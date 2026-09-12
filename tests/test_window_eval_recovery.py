import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "window_recovery", ROOT / "scripts/recover_window_evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(tmp_path, mod):
    root = tmp_path / "runs"
    training = tmp_path / "training"
    analysis = tmp_path / "analysis"
    training.mkdir()
    analysis.mkdir()
    (training / "DEPLOYED_COMMIT").write_text(mod.TRAINING_COMMIT)
    (analysis / "ANALYSIS_COMMIT").write_text("a" * 40)
    for variant, mode in [("random3", "random"), ("sliding3", "sliding")]:
        run = root / variant
        run.mkdir(parents=True)
        (run / "exit_code.txt").write_text("0\n")
        (run / "run_card.json").write_text(json.dumps({
            "variant": variant, "seed": 21, "total_training_steps": 200,
            "opd_window_mode": mode, "opd_window_seed": 910021,
            "source_commit": mod.TRAINING_COMMIT}))
        (run / "checkpoint_acceptance.json").write_text(json.dumps({
            "passed": True, "issues": [], "checkpoint_steps": [50, 100, 200]}))
        (run / "window_acceptance.json").write_text(json.dumps({
            "passed": True, "issues": [], "last_step": 200}))
    for step in [50, 100, 200]:
        d = root / "random3" / f"eval_step_{step}_n8"
        (d / "outputs").mkdir(parents=True)
        (d / "exit_code.txt").write_text("0\n")
        (d / "outputs/example.jsonl").write_text('{"response":"untouched"}\n')
    failed = root / "random3/queue_jobs/full_audit"
    failed.mkdir(parents=True)
    (failed / "exit_code.txt").write_text("1\n")
    (failed / "evidence.txt").write_text("original failure")
    (root / "queue_state.json").write_text(json.dumps({"status": "failed", "job": str(failed)}))
    return root, training, analysis


def test_recovery_preserves_failure_and_hashes_existing_evidence(tmp_path):
    mod = module()
    root, training, analysis = fixture(tmp_path, mod)
    old_state = (root / "queue_state.json").read_bytes()
    record = mod.prepare_recovery(root, training, analysis, "jsonl_r1")
    assert record["training_commit"] == mod.TRAINING_COMMIT
    assert record["analysis_commit"] == "a" * 40
    assert (root / "recoveries/jsonl_r1/queue_state.json").read_bytes() == old_state
    assert (root / "random3/queue_jobs/full_audit/evidence.txt").read_text() == "original failure"
    mod.verify_protected(record)
    with pytest.raises(FileExistsError):
        mod.prepare_recovery(root, training, analysis, "jsonl_r1")


def test_recovery_refuses_unfinished_training_or_started_sliding_eval(tmp_path):
    mod = module()
    root, training, analysis = fixture(tmp_path, mod)
    p = root / "sliding3/exit_code.txt"
    p.write_text("1\n")
    with pytest.raises(ValueError, match="training"):
        mod.prepare_recovery(root, training, analysis, "jsonl_r1")
    assert not (root / "recoveries/jsonl_r1").exists()
    p.write_text("0\n")
    (root / "sliding3/eval_step_50_n8").mkdir()
    with pytest.raises(ValueError, match="Sliding3"):
        mod.prepare_recovery(root, training, analysis, "jsonl_r1")


def test_recovery_refuses_invalid_identifier_and_changed_training_runtime(tmp_path):
    mod = module()
    root, training, analysis = fixture(tmp_path, mod)
    with pytest.raises(ValueError, match="identifier"):
        mod.prepare_recovery(root, training, analysis, "../unsafe")
    (training / "DEPLOYED_COMMIT").write_text("b" * 40)
    with pytest.raises(ValueError, match="training runtime"):
        mod.prepare_recovery(root, training, analysis, "jsonl_r1")


def test_protected_evidence_change_is_not_silently_accepted(tmp_path):
    mod = module()
    root, training, analysis = fixture(tmp_path, mod)
    record = mod.prepare_recovery(root, training, analysis, "jsonl_r1")
    (root / "random3/eval_step_50_n8/outputs/example.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="protected"):
        mod.verify_protected(record)


def test_execution_only_generates_missing_sliding_evals_with_frozen_code(tmp_path, monkeypatch):
    mod = module()
    root, training, analysis = fixture(tmp_path, mod)
    card_path = root / "sliding3/run_card.json"
    card = json.loads(card_path.read_text())
    card["student_model"] = "/models/Qwen3-1.7B-Base"
    card_path.write_text(json.dumps(card))
    data = tmp_path / "data"
    data.mkdir()
    for task in mod.queue.TASKS:
        (data / f"{task}.jsonl").write_text("{}\n")
    calls = []
    monkeypatch.setattr(mod.queue, "run_job", lambda argv, *a, **kw: calls.append((list(map(str, argv)), a, kw)))
    mod.execute_recovery(root, training, analysis, root / "recoveries/r1",
                         "/env/bin/python", data, tmp_path / "grader.py", {})
    assert all("command.sh" not in " ".join(argv) for argv, _, _ in calls)
    generations = [(argv, kw) for argv, _, kw in calls if "eval_qwen3_math_vllm.py" in argv[1]]
    assert len(generations) == 3
    for argv, kw in generations:
        assert argv[1] == str(training / "scripts/eval_qwen3_math_vllm.py")
        assert "sliding3" in argv[argv.index("--output-dir") + 1]
        assert argv[argv.index("--n") + 1] == "8"
        assert argv[argv.index("--grader") + 1] == "verl"
        assert kw["gpu"] is True
    regrades = [argv for argv, _, _ in calls if "regrade_opd_eval_external.py" in argv[1]]
    assert len(regrades) == 2
    assert all(argv[1] == str(analysis / "scripts/regrade_opd_eval_external.py") for argv in regrades)
    assert all(argv[-2:] == ["--steps", "50,100,200"] for argv in regrades)
    comparisons = [argv for argv, _, _ in calls if "compare_paired_opd_evals.py" in argv[1]]
    assert len(comparisons) == 1
    assert comparisons[0][comparisons[0].index("--bootstrap-replicates") + 1] == "10000"
