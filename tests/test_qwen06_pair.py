import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts/run_qwen06_pair.py"
    assert path.is_file(), "Qwen06 paired queue is not implemented"
    spec = importlib.util.spec_from_file_location("qwen06_pair", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cards(mod):
    base = {**mod.paired.EXPECTED_TRAINING_VALUES,
            "student_model": str(mod.assets.STUDENT), "student_model_revision": mod.assets.REVISION,
            "teacher_model": str(mod.assets.TEACHER), "source_commit": "a" * 40,
            "ppo_epochs": 1, "opd_window_mode": "fixed"}
    return {v: {**base, "variant": v, "opd_block_size": k, "opd_block_advantage_mode": mode}
            for v, k, mode in (("token_opd", 1, "sum"), ("block3_mean", 3, "mean"))}


def test_approved_scope_and_identical_control():
    mod = module()
    assert mod.VARIANTS == ("token_opd", "block3_mean")
    assert mod.BUDGET_SECONDS == 48 * 3600
    assert mod.STEPS == (50, 100, 200)
    mod.validate_prepared_pair(cards(mod))


@pytest.mark.parametrize("key,value", [("seed", 22), ("learning_rate", 1e-6), ("ppo_epochs", 2),
    ("student_model", "Qwen3-4B"), ("opd_window_mode", "sliding"), ("opd_block_size", 5),
    ("rollout_top_p", 0.95), ("unlisted_setting", "changed")])
def test_pair_rejects_any_unapproved_difference(key, value):
    mod = module()
    pair = cards(mod)
    pair["block3_mean"][key] = value
    with pytest.raises(ValueError):
        mod.validate_prepared_pair(pair)


def test_predecessor_must_finish_all_jobs_and_protected_input_audit(tmp_path):
    mod = module()
    state = tmp_path / "queue_state.json"
    state.write_text(json.dumps({"status": "running"}))
    assert not mod.predecessor_ready(tmp_path)
    state.write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="predecessor"):
        mod.predecessor_ready(tmp_path)
    state.write_text(json.dumps({"status": "complete", "recovery_id": "x"}))
    with pytest.raises(ValueError, match="predecessor"):
        mod.predecessor_ready(tmp_path)
    proof = tmp_path / "recoveries/x/protected_inputs_verified.json"
    proof.parent.mkdir(parents=True)
    proof.write_text(json.dumps({"passed": True}))
    (tmp_path / "paired_comparison.json").write_text("{}")
    for variant in ("random3", "sliding3"):
        for step in mod.STEPS:
            exit_file = tmp_path / variant / f"eval_step_{step}_n8/exit_code.txt"
            exit_file.parent.mkdir(parents=True)
            exit_file.write_text("0")
    assert mod.predecessor_ready(tmp_path)


@pytest.mark.parametrize("remote", ["train", "ml2"])
def test_local_transport_rejects_unapproved_target_without_ssh(tmp_path, remote):
    fake = tmp_path / "ssh"
    fake.write_text("#!/bin/sh\nprintf unexpected_ssh >&2\nexit 42\n")
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "REMOTE": remote,
           "REMOTE_ROOT": "/tmp/invalid", "LAUNCH_TRANSPORT": "local", "PREPARE_ONLY": "true"}
    result = subprocess.run(["bash", str(ROOT / "scripts/launch_revisiting_block_opd_formal_train.sh")],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "local transport requires approved ml2 runtime" in result.stderr


def test_launcher_env_locks_both_variants_and_probe_resume(tmp_path):
    mod = module()
    env = mod.launcher_env(tmp_path, "a" * 40, tmp_path / "runs", "token_opd")
    other = mod.launcher_env(tmp_path, "a" * 40, tmp_path / "runs", "block3_mean")
    assert {k for k in env if env[k] != other[k]} == {"VARIANT", "EXP_NAME", "OPD_DIAG_OUTPUT_DIR"}
    assert env["OPD_DIAG_OUTPUT_DIR"] == str(tmp_path / "runs/token_opd/diagnostics")
    assert env["TOTAL_TRAINING_STEPS"] == "200"
    assert env["DIAGNOSTIC_SAVE_STEPS"] == "50,100,200"
    assert env["STUDENT_MODEL"] == str(mod.assets.STUDENT)
    assert env["REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"] == "1"
    assert env["TEST_FREQ"] == "-1" and env["RESUME_MODE"] == "disable"
    probe = mod.launcher_env(tmp_path, "a" * 40, tmp_path / "probes", "token_opd", probe_step=2)
    assert probe["TOTAL_TRAINING_STEPS"] == "2"
    assert probe["RESUME_MODE"] == "resume_path"
    assert probe["RESUME_FROM_PATH"].endswith("token_opd/checkpoints/global_step_1")
    assert probe["MAX_RESPONSE_LENGTH"] == env["MAX_RESPONSE_LENGTH"]
    assert probe["OPD_DIAG_OUTPUT_DIR"] == str(tmp_path / "probes/token_opd/diagnostics")


def test_completion_refuses_expired_deadline(monkeypatch):
    mod = module()
    monkeypatch.setattr(mod.time, "time", lambda: 200)
    with pytest.raises(TimeoutError, match="budget"):
        mod.require_budget(199)


def test_audit_command_requires_public_student_and_full_protocol(tmp_path):
    mod = module()
    command = list(map(str, mod.audit_command(tmp_path, tmp_path / "token_opd", "a" * 40, full=True)))
    assert command[command.index("--expected-student-model-revision") + 1] == mod.assets.REVISION
    assert command[command.index("--eval-steps") + 1] == "50,100,200"
    probe = list(map(str, mod.audit_command(tmp_path, tmp_path / "token_opd", "a" * 40, probe_step=2)))
    assert probe[probe.index("--expected-resume-mode") + 1] == "resume_path"
    assert "--skip-eval" in probe


def test_execution_order_token_full_eval_before_block_and_no_extra_runs(tmp_path, monkeypatch):
    mod = module()
    calls = []
    monkeypatch.setattr(mod, "prepare", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "check_pair_artifacts", lambda *a: None)

    def runner(argv, job, **kwargs):
        args = list(map(str, argv))
        calls.append((args, job, kwargs))
        if args[0] == "bash":
            logs = job / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / kwargs.get("log_name", "job.log")).write_text("probe log")

    mod.execute_pair(tmp_path, tmp_path / "runtime", "a" * 40, runner)
    gpu = [args for args, _, kw in calls if kw.get("gpu")]
    assert len(gpu) == 12  # two probe launches + training + three evals, per variant
    assert all("token_opd" in " ".join(args) for args in gpu[:6])
    assert all("block3_mean" in " ".join(args) for args in gpu[6:])
    evals = [a for a in gpu if "--model-path" in a]
    assert len(evals) == 6
    assert all(a[a.index("--n") + 1] == "8" for a in evals)
    assert all(a[a.index("--max-tokens") + 1] == "16384" for a in evals)
    assert sum("--grader-source" in a for a, _, _ in calls) == 2


def test_resume_state_requires_scheduler_rng_and_consumed_data():
    mod = module()
    good = {"lr_scheduler": {"last_epoch": 1},
            "rng": {"cpu": [1], "cuda": [2], "numpy": [3], "random": [4]}}
    mod.validate_resume_state(good, {"_num_yielded": 1}, 1)
    with pytest.raises(ValueError, match="RNG"):
        mod.validate_resume_state({**good, "rng": {"cpu": [1]}}, {"_num_yielded": 1}, 1)
    with pytest.raises(ValueError, match="scheduler"):
        mod.validate_resume_state({**good, "lr_scheduler": {"last_epoch": 0}}, {"_num_yielded": 1}, 1)
    with pytest.raises(ValueError, match="dataloader"):
        mod.validate_resume_state(good, {}, 1)
