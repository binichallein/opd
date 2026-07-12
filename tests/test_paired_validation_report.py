import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_paired_validation_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("paired_validation_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def comparison(view: str, *, source_commit: str = "abc123"):
    steps = {}
    for step, delta in ((50, 0.01), (100, 0.02), (200, 0.03)):
        steps[str(step)] = {
            "step": step,
            "pair_count": 643,
            "rollout_key_count": 5144,
            "macro": {
                "avg_at_n": {"left": 0.2, "right": 0.2 + delta, "delta": delta},
                "pass_at_n": {"left": 0.3, "right": 0.3 + delta, "delta": delta},
            },
            "bootstrap": {
                "avg_at_n": {"lower_95": 0.001, "upper_95": 0.05},
                "pass_at_n": {"lower_95": 0.002, "upper_95": 0.06},
            },
            "prompt_outcomes": {
                "avg_at_n": {"win": 100, "tie": 500, "loss": 43},
                "pass_at_n": {"win": 80, "tie": 530, "loss": 33},
            },
            "per_task": {
                task: {
                    "avg_at_n": {"left": 0.2, "right": 0.23, "delta": 0.03},
                    "pass_at_n": {"left": 0.3, "right": 0.33, "delta": 0.03},
                }
                for task in ("math500", "aime24", "aime25", "amc23")
            },
        }
    return {
        "left": {"label": "token_opd", "run_dir": "/runs/token"},
        "right": {"label": "block3_mean", "run_dir": "/runs/block3"},
        "training_contract": {
            "left_variant": "token_opd",
            "right_variant": "block3_mean",
            "source_commit": source_commit,
            "student_model": "/models/DeepSeek-R1-Distill-Qwen-1.5B",
            "student_model_revision": "student-rev",
            "teacher_model": "/models/JustRL-DeepSeek-1.5B",
            "teacher_model_revision": "teacher-rev",
            "train_sha256": "a" * 64,
            "test_sha256": "b" * 64,
            "prompt_batch_hashes_matched": 41,
        },
        "grader_view": view,
        "bootstrap": {
            "method": "task-stratified paired prompt bootstrap",
            "replicates": 10000,
            "seed": 20260712,
        },
        "steps": steps,
        "decision": {
            "step": 200,
            "status": "strong_single_seed_support",
            "scope": "one paired training seed; bootstrap does not estimate training-run variance",
        },
    }


def test_report_renders_verdict_contract_metrics_and_relative_heatmaps(tmp_path):
    module = load_module()
    external = comparison("historical_external_grader")
    builtin = comparison("outputs")
    token_diagnostics = tmp_path / "assets" / "token"
    block_diagnostics = tmp_path / "assets" / "block3"
    token_diagnostics.mkdir(parents=True)
    block_diagnostics.mkdir(parents=True)
    for directory in (token_diagnostics, block_diagnostics):
        for name in module.DIAGNOSTIC_IMAGES:
            (directory / name).write_bytes(b"png")
    output = tmp_path / "report" / "paired.html"

    module.write_report(
        external,
        builtin,
        output,
        token_diagnostics=token_diagnostics,
        block_diagnostics=block_diagnostics,
        title="DeepSeek / JustRL Block3 跨师生复现",
    )

    text = output.read_text(encoding="utf-8")
    assert "强复现（单训练 seed）" in text
    assert "DeepSeek-R1-Distill-Qwen-1.5B" in text
    assert "JustRL-DeepSeek-1.5B" in text
    assert "0.0300" in text
    assert "[0.0010, 0.0500]" in text
    assert "41" in text
    assert "../assets/token/position_heatmaps.png" in text
    assert "../assets/block3/position_heatmaps.png" in text
    assert str(tmp_path) not in text
    assert ".contract { width:100%; min-width:0; table-layout:fixed;" in text


def test_report_rejects_builtin_and_external_training_contract_mismatch(tmp_path):
    module = load_module()
    external = comparison("historical_external_grader")
    builtin = comparison("outputs", source_commit="different")

    with pytest.raises(ValueError, match="training contracts differ"):
        module.write_report(
            external,
            builtin,
            tmp_path / "report.html",
            token_diagnostics=tmp_path / "token",
            block_diagnostics=tmp_path / "block",
            title="test",
        )


def test_report_cli_reads_comparison_json(tmp_path):
    module = load_module()
    external_path = tmp_path / "external.json"
    builtin_path = tmp_path / "builtin.json"
    external_path.write_text(json.dumps(comparison("historical_external_grader")))
    builtin_path.write_text(json.dumps(comparison("outputs")))

    assert module.load_json(external_path)["grader_view"] == "historical_external_grader"
    assert module.load_json(builtin_path)["grader_view"] == "outputs"
