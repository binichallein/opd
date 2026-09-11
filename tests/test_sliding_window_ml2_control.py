import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONTROL = SCRIPTS / "sliding_window_ml2_control.sh"
LAUNCHER = SCRIPTS / "launch_revisiting_block_opd_formal_train.sh"
RUNNER = SCRIPTS / "run_revisiting_sampled_block_opd_math.sh"
COMMIT = "b" * 40


@pytest.fixture
def captured_ssh(tmp_path):
    capture = tmp_path / "ssh.jsonl"
    fake = tmp_path / "ssh"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['SSH_CAPTURE'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if os.environ.get('FAIL_AUDIT') and 'audit_window_control.py' in sys.argv[-1]:\n"
        "    sys.exit(19)\n"
    )
    fake.chmod(0o755)
    env = {
        "HOME": os.environ["HOME"],
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SSH_CAPTURE": str(capture),
        "SOURCE_COMMIT": COMMIT,
    }
    return env, capture


def run_shell(script, args, env):
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def ssh_calls(capture):
    return [json.loads(line) for line in capture.read_text().splitlines()]


def run_card(command):
    match = re.search(r"cat > '[^']+/run_card.json' <<JSON\n(.*?)\nJSON", command, re.S)
    assert match, command
    return json.loads(match[1])


@pytest.mark.parametrize("variant,mode", [("random3", "random"), ("sliding3", "sliding")])
def test_window_wrapper_locks_ml2_contract_and_records_provenance(captured_ssh, variant, mode):
    env, capture = captured_ssh
    result = run_shell(
        CONTROL,
        [variant, "formal"],
        {
            **env,
            "REMOTE": "train",
            "HOST_TAG": "train",
            "DATE_TAG": "old",
            "RUN_TAG": "token_opd_replication",
            "RUNTIME_ROOT": "/old/runtime",
            "REMOTE_ROOT": "/old/runtime",
            "CACHE_ROOT": "/old/cache",
            "ENV_SEED": "99",
            "OPD_WINDOW_SEED": "4",
            "OPD_WINDOW_MODE": "fixed",
            "AUDIT_SCRIPT_OVERRIDE": "/old/audit_block10_run.py",
        },
    )
    assert result.returncode == 0, result.stderr
    calls = ssh_calls(capture)
    assert all(call[0] == "ml2" for call in calls)
    audits = [
        call[-1]
        for call in calls
        if "audit_window_control.py" in call[-1] and "nohup bash" not in call[-1]
    ]
    assert audits and "--checkpoint-steps 1,2" in audits[0]
    command = next(call[-1] for call in calls if "nohup bash" in call[-1])
    card = run_card(command)
    assert card["variant"] == variant
    assert card["opd_window_mode"] == mode
    assert card["opd_window_seed"] == 910021
    assert card["opd_block_size"] == 3
    assert card["opd_block_advantage_mode"] == "mean"
    assert card["ppo_epochs"] == 1
    assert card["seed"] == 21
    assert card["source_commit"] == COMMIT
    assert card["total_training_steps"] == 200
    assert card["diagnostic_save_steps"] == "50,100,200"
    assert card["save_freq"] == -1
    assert card["resume_mode"] == "disable"
    assert "no automatic deletion" in card["checkpoint_policy"]
    assert f"deployments/{COMMIT}" in command
    assert f"20260911v2_sliding_window_seed21_ml2/{variant}" in command
    assert f"/swr/0911v2/{variant}/formal" in command
    assert "OPD_WINDOW_SEED='910021'" in command
    assert "opd_ext/window_supervision.py" in command
    assert "window_supervision_sha256" in card
    assert "9b3b8b76" not in command


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["token_opd"],
        ["block3_mean", "formal"],
        ["random10"],
        ["random3", "bad-action"],
        ["random3", "eval", "75"],
        ["sliding3", "formal", "unexpected"],
    ],
)
def test_unsupported_requests_fail_before_ssh(captured_ssh, args):
    env, capture = captured_ssh
    result = run_shell(CONTROL, args, env)
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert not capture.exists()


def test_default_action_is_read_only_status_and_source_defaults_to_head(captured_ssh):
    env, capture = captured_ssh
    env.pop("SOURCE_COMMIT")
    result = run_shell(CONTROL, ["sliding3"], env)
    assert result.returncode == 0, result.stderr
    commands = ssh_calls(capture)
    assert len(commands) == 1 and commands[0][0] == "ml2"
    assert "20260911v2_sliding_window_seed21_ml2/sliding3" in commands[0][-1]
    assert "nohup bash" not in commands[0][-1]


def test_wrapper_uses_head_when_source_commit_is_unset(captured_ssh):
    env, capture = captured_ssh
    env.pop("SOURCE_COMMIT")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = run_shell(CONTROL, ["random3", "probe1"], env)
    assert result.returncode == 0, result.stderr
    command = next(call[-1] for call in ssh_calls(capture) if "nohup bash" in call[-1])
    assert run_card(command)["source_commit"] == head


@pytest.mark.parametrize("action", ["probe2", "formal", "eval"])
def test_window_audit_failure_blocks_launch(captured_ssh, action):
    env, capture = captured_ssh
    result = run_shell(CONTROL, ["random3", action], {**env, "FAIL_AUDIT": "1"})
    assert result.returncode == 19, result.stderr
    assert not any("nohup bash" in call[-1] for call in ssh_calls(capture))


@pytest.mark.parametrize(
    "action,steps",
    [("probe-audit", "1,2"), ("audit-checkpoints", "50,100,200"), ("audit", "50,100,200")],
)
def test_all_audit_actions_use_window_adapter(captured_ssh, action, steps):
    env, capture = captured_ssh
    result = run_shell(CONTROL, ["sliding3", action], env)
    assert result.returncode == 0, result.stderr
    audit = next(call[-1] for call in ssh_calls(capture) if "audit_window_control.py" in call[-1])
    assert f"--checkpoint-steps {steps}" in audit
    assert "--variant sliding3" in audit
    assert f"deployments/{COMMIT}/scripts/audit_window_control.py" in audit
    if action == "audit":
        assert "--eval-steps 50,100,200" in audit


@pytest.mark.parametrize("step", [50, 100, 200])
def test_eval_preserves_full_benchmarks_and_sample_counts(captured_ssh, step):
    env, capture = captured_ssh
    result = run_shell(CONTROL, ["random3", "eval", str(step)], env)
    assert result.returncode == 0, result.stderr
    command = next(call[-1] for call in ssh_calls(capture) if "eval_card.json" in call[-1])
    assert "math500 aime24 aime25 amc23" in command
    assert "--n '8'" in command
    assert "--limit" not in command
    assert f"global_step_{step}" in command


@pytest.mark.parametrize(
    "variant,mode,k,adv",
    [
        ("random3", "random", 3, "mean"),
        ("sliding3", "sliding", 3, "mean"),
        ("token_opd", "fixed", 1, "sum"),
        ("block3_sum", "fixed", 3, "sum"),
        ("block3_mean", "fixed", 3, "mean"),
        ("block5_mean", "fixed", 5, "mean"),
        ("block10_mean", "fixed", 10, "mean"),
        ("block3_mixed_lam05", "fixed", 3, "mixed"),
    ],
)
def test_runner_sets_declared_hydra_window_fields_and_preserves_legacy(
    tmp_path,
    variant,
    mode,
    k,
    adv,
):
    root = tmp_path / "runtime"
    (root / "scripts").mkdir(parents=True)
    (root / "external/revisiting_opd").mkdir(parents=True)
    shutil.copy(RUNNER, root / "scripts" / RUNNER.name)
    fake = tmp_path / "python3"
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    fake.chmod(0o755)
    result = run_shell(
        root / "scripts" / RUNNER.name,
        [],
        {
            "HOME": os.environ["HOME"],
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "VARIANT": variant,
            "ENV_SEED": "22",
            "OPD_WINDOW_SEED": "910022",
            "OPD_WINDOW_MODE": "random",
            "LOG_DIR": str(tmp_path / "logs"),
        },
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert f"actor_rollout_ref.actor.opd_window_mode={mode}" in args
    assert "actor_rollout_ref.actor.opd_window_seed=910022" in args
    assert f"actor_rollout_ref.actor.opd_block_size={k}" in args
    assert f"actor_rollout_ref.actor.opd_block_advantage_mode={adv}" in args
    assert "actor_rollout_ref.actor.ppo_epochs=1" in args


def test_formal_launcher_keeps_legacy_window_defaults(captured_ssh):
    env, capture = captured_ssh
    result = run_shell(LAUNCHER, [], {**env, "VARIANT": "block3_mean", "ENV_SEED": "22"})
    assert result.returncode == 0, result.stderr
    card = run_card(ssh_calls(capture)[0][-1])
    assert card["opd_window_mode"] == "fixed"
    assert card["opd_window_seed"] == 910022


def test_sync_and_hash_evidence_include_all_window_dependencies():
    sync = (SCRIPTS / "sync_block3_replication_to_ml2.sh").read_text()
    launcher = LAUNCHER.read_text()
    for dependency in (
        "opd_ext/window_supervision.py",
        "scripts/audit_window_run.py",
        "scripts/sliding_window_ml2_control.sh",
        "scripts/audit_window_control.py",
    ):
        assert dependency in sync
        assert dependency in launcher


def test_sync_includes_queue_and_full_acceptance_dependencies():
    sync = (SCRIPTS / "sync_block3_replication_to_ml2.sh").read_text()
    for dependency in (
        "scripts/run_window_queue.py",
        "scripts/regrade_opd_eval_external.py",
        "scripts/compare_paired_opd_evals.py",
        "scripts/check_final_acceptance.py",
        "opd_ext/analysis.py",
    ):
        assert dependency in sync


def test_sync_includes_isolated_window_plotting_runtime():
    sync = (SCRIPTS / "sync_block3_replication_to_ml2.sh").read_text()
    assert "scripts/plot_window_diagnostics.py" in sync
    assert "configs/window_plot_requirements.txt" in sync


def test_prepare_only_is_opt_in_and_before_nohup(captured_ssh, tmp_path):
    env, capture = captured_ssh
    run_root = tmp_path / "runs"
    result = run_shell(
        LAUNCHER,
        [],
        {
            **env,
            "VARIANT": "random3",
            "PREPARE_ONLY": "true",
            "RUN_ROOT": str(run_root),
        },
    )
    assert result.returncode == 0, result.stderr
    command = ssh_calls(capture)[0][-1]
    assert 'PREPARE_ONLY="${PREPARE_ONLY:-false}"' in LAUNCHER.read_text()
    assert "bash -n" in command and "-m json.tool" in command
    assert command.index("prepared command=") < command.index("nohup bash")
    run_dir = run_root / "random3"
    (run_dir / "logs").mkdir(parents=True)
    marker = tmp_path / "launched"
    (run_dir / "command.sh").write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    # Execute only the final launch gate locally, never the remote setup payload.
    suffix = command[command.index(f"chmod +x '{run_dir}/command.sh'") :]
    prepared = subprocess.run(["bash", "-c", suffix], env=env, text=True, capture_output=True)
    assert prepared.returncode == 0, prepared.stderr
    assert f"prepared command={run_dir}/command.sh" in prepared.stdout
    assert not marker.exists()
    assert not (run_dir / "train.pid").exists()


def test_prepare_only_formal_still_requires_window_resume_gate(captured_ssh):
    env, capture = captured_ssh
    result = run_shell(
        CONTROL,
        ["sliding3", "formal"],
        {
            **env,
            "PREPARE_ONLY": "true",
            "FAIL_AUDIT": "1",
        },
    )
    assert result.returncode == 19, result.stderr
    assert not any("run_card.json' <<JSON" in call[-1] for call in ssh_calls(capture))


def test_prepare_only_rejects_invalid_boolean_before_ssh(captured_ssh):
    env, capture = captured_ssh
    result = run_shell(LAUNCHER, [], {**env, "PREPARE_ONLY": "yes"})
    assert result.returncode == 2
    assert not capture.exists()


def test_prepare_only_inherits_through_formal_wrapper_after_successful_audits(captured_ssh):
    env, capture = captured_ssh
    result = run_shell(CONTROL, ["sliding3", "formal"], {**env, "PREPARE_ONLY": "true"})
    assert result.returncode == 0, result.stderr
    commands = [call[-1] for call in ssh_calls(capture)]
    payload = next(command for command in commands if "run_card.json' <<JSON" in command)
    assert "if [[ 'true' == true ]]" in payload
    assert any(
        "audit_window_control.py" in command for command in commands[: commands.index(payload)]
    )


def test_default_launcher_still_reaches_nohup_gate(captured_ssh):
    env, capture = captured_ssh
    result = run_shell(LAUNCHER, [], env)
    assert result.returncode == 0, result.stderr
    assert "if [[ 'false' == true ]]" in ssh_calls(capture)[0][-1]


@pytest.mark.parametrize("source", ["HEAD", "main", "abc123", "invalid;command"])
def test_wrapper_rejects_nonimmutable_source_without_ssh(captured_ssh, source):
    env, capture = captured_ssh
    result = run_shell(CONTROL, ["random3"], {**env, "SOURCE_COMMIT": source})
    assert result.returncode == 2
    assert "immutable" in result.stderr
    assert not capture.exists()


@pytest.mark.parametrize(
    "name", [CONTROL.name, LAUNCHER.name, RUNNER.name, "sync_block3_replication_to_ml2.sh"]
)
def test_shell_syntax(name):
    result = subprocess.run(["bash", "-n", str(SCRIPTS / name)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
