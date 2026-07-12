import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_block3_ml2_replication_report.py"
TRAIN_SHA = "a" * 64
EVAL_SHA = "b" * 64


def load_module():
    spec = importlib.util.spec_from_file_location("block3_ml2_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def task_summary(avg=0.2, passed=0.3):
    examples = {"math500": 500, "aime24": 30, "aime25": 30, "amc23": 83}
    return {
        task: {
            "num_examples": count,
            "total_rollouts": count * 8,
            "avg_at_n": avg,
            "pass_at_n": passed,
            "format_error_rollouts": 1,
            "avg_response_length_tokens": 100.0,
        }
        for task, count in examples.items()
    }


def eval_summary(avg=0.2, passed=0.3, grader="verl"):
    return {
        "n_expected": 8,
        "grader": grader,
        "eval_seed": 21,
        "rollout_seeds": list(range(21, 29)),
        "enable_thinking": False,
        "tasks": task_summary(avg, passed),
        "macro_avg_at_n": avg,
        "macro_pass_at_n": passed,
    }


def legacy_summary(avg, passed):
    tasks = {
        task: {
            "avg_at_8": avg,
            "pass_at_8": passed,
            "format_errors": 1,
            "avg_length_tokens": 100.0,
        }
        for task in ("math500", "aime24", "aime25", "amc23")
    }
    return {"macro_avg_at_8": avg, "macro_pass_at_8": passed, "tasks": tasks}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_fixture(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "run_card.json",
        {
            "variant": "block3_mean",
            "source_commit": "abc123",
            "experiment_name": "block3-mean-replication-ml2-test",
            "student_model": "/models/Qwen3-1.7B-Base",
            "teacher_model": "/models/Qwen3-4B-Base-GRPO",
            "train_data": "/data/train.parquet",
            "expected_train_sha256": TRAIN_SHA,
            "seed": 21,
            "total_training_steps": 200,
            "train_batch_size": 4,
            "rollout_group_size": 8,
            "learning_rate": 2e-6,
            "max_response_length": 16384,
            "actor_ppo_micro_batch_size_per_gpu": 1,
            "ref_log_prob_micro_batch_size_per_gpu": 1,
            "rollout_log_prob_micro_batch_size_per_gpu": 4,
            "rollout_gpu_memory_utilization": 0.6,
            "diagnostic_output_dir": "/authoritative/run/diagnostics",
        },
    )
    write_json(
        run_dir / "acceptance.json",
        {
            "passed": True,
            "checkpoint_steps": [50, 100, 200],
            "diagnostic_steps": [1, *range(5, 201, 5)],
            "eval_steps": [50, 100, 200],
            "source_commit": "abc123",
            "issues": [],
            "warnings": [],
        },
    )
    (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    (run_dir / "env.txt").write_text(
        "0, NVIDIA A100-SXM4-80GB, 81920 MiB\n"
        "1, NVIDIA A100-SXM4-80GB, 81920 MiB\n"
        "2, NVIDIA A100-SXM4-80GB, 81920 MiB\n"
        "3, NVIDIA A100-SXM4-80GB, 81920 MiB\n",
        encoding="utf-8",
    )
    write_json(
        run_dir / "data_manifest.json",
        {
            "counts": {"train": 1_791_700},
            "sha256": {
                "train_parquet": TRAIN_SHA,
                "test_parquet": EVAL_SHA,
            },
        },
    )
    records = []
    for step in [1, *range(5, 201, 5)]:
        records.append(
            {
                "step": step,
                "diagnostics/student_entropy": step / 100,
                "diagnostics/teacher_entropy": step / 120,
                "diagnostics/topk_overlap_ratio": 0.6,
                "diagnostics/student_overlap_mass": 0.9,
                "diagnostics/teacher_overlap_mass": 0.9,
                "diagnostics/weighted_sign_flip_rate": 0.1,
                "diagnostics/normalized_leakage": 1.0,
                "actor/grad_norm": 2.0,
                "diagnostics/post_update_block_ratio_outside_clip_fraction": 0.01,
                "response_length/clip_ratio": 0.0,
                "diagnostics/student_entropy_nonfinite_count": 0,
                "diagnostics/teacher_entropy_nonfinite_count": 0,
                "diagnostics/raw_token_advantage_nonfinite_count": 0,
                "diagnostics/block_advantage_nonfinite_count": 0,
                "diagnostics/post_update_block_ratio_overflow_count": 0,
                "diagnostics/post_update_block_ratio_underflow_count": 0,
                "diagnostics/post_update_block_ratio_nonfinite_count": 0,
            }
        )
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True)
    (diagnostics / "scalars.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    for step, score in ((50, 0.10), (100, 0.08), (200, 0.14)):
        write_json(
            run_dir / f"eval_step_{step}_n8" / "outputs" / "summary.json",
            eval_summary(score, score + 0.1),
        )
        write_json(
            run_dir / f"eval_step_{step}_n8" / "historical_external_grader" / "summary.json",
            eval_summary(score + 0.05, score + 0.15, grader="external"),
        )
        write_json(
            run_dir / f"eval_step_{step}_n8" / "outputs" / "eval_config.json",
            {
                "tasks": list(("math500", "aime24", "aime25", "amc23")),
                "n": 8,
                "temperature": 1.0,
                "top_p": 0.9,
                "max_tokens": 16384,
                "eval_seed": 21,
                "rollout_seeds": list(range(21, 29)),
                "grader": "verl",
                "enable_thinking": False,
            },
        )
        hashes = run_dir / f"eval_step_{step}_n8" / "historical_external_grader" / "input_hashes.sha256"
        hash_lines = [
            f"{'1' * 64}  /run/eval_step_{step}_n8/outputs/{task}_t1.0_p0.9_n8-MNT16384.jsonl"
            for task in ("math500", "aime24", "aime25", "amc23")
        ]
        hash_lines.append(
            "04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f  "
            "/run/grading/historical_utils_sha04f7.py"
        )
        hashes.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    historical = tmp_path / "historical.json"
    write_json(
        historical,
        {
            "training": {
                "student": "Qwen3-1.7B-Base",
                "teacher": "Qwen3-4B-Base-GRPO",
                "train_sha256": TRAIN_SHA,
                "eval_sha256": EVAL_SHA,
                "seed": 21,
                "steps": 200,
                "train_batch_size": 4,
                "rollouts_per_prompt": 8,
                "max_response_tokens": 16384,
                "learning_rate": 2e-6,
            },
            "evaluation": {
                "n": 8,
                "temperature": 1.0,
                "top_p": 0.9,
                "max_tokens": 16384,
                "tasks": list(("math500", "aime24", "aime25", "amc23")),
            },
            "comparability": {
                "hardware": {
                    "token_block3_block5": "4x NVIDIA A800-SXM4-80GB"
                }
            },
            "results": {
                "token_opd": legacy_summary(0.30, 0.40),
                "block3_mean": legacy_summary(0.31, 0.43),
            }
        },
    )
    return run_dir, historical


def test_result_keeps_grader_views_separate_and_rejects_causal_replication(tmp_path):
    module = load_module()
    run_dir, historical = make_fixture(tmp_path)

    result = module.build_result(run_dir, historical)

    assert result["status"] == "complete"
    assert result["run"]["run_dir"] == "/authoritative/run"
    assert result["contract"]["training_data_sha256"] == TRAIN_SHA
    assert result["evaluation"]["fixed_seed_builtin_verl"]["200"]["grader"] == "verl"
    assert (
        result["evaluation"]["fixed_seed_historical_external_grader"]["200"]["grader"]
        == "external"
    )
    assert result["comparability"]["same_host_ml2_token_baseline_available"] is False
    assert result["conclusion"]["status"] == "not_independently_validated"


def test_result_rejects_failed_acceptance(tmp_path):
    module = load_module()
    run_dir, historical = make_fixture(tmp_path)
    acceptance_path = run_dir / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["passed"] = False
    acceptance["issues"] = ["checkpoint 200 missing"]
    write_json(acceptance_path, acceptance)

    with pytest.raises(ValueError, match="acceptance audit failed"):
        module.build_result(run_dir, historical)


def test_result_rejects_external_grader_contract_mismatch(tmp_path):
    module = load_module()
    run_dir, historical = make_fixture(tmp_path)
    summary_path = (
        run_dir
        / "eval_step_200_n8"
        / "historical_external_grader"
        / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["n_expected"] = 1
    write_json(summary_path, summary)

    with pytest.raises(ValueError, match="external grader.*n=8"):
        module.build_result(run_dir, historical)


def test_result_rejects_wrong_training_variant(tmp_path):
    module = load_module()
    run_dir, historical = make_fixture(tmp_path)
    run_card_path = run_dir / "run_card.json"
    run_card = json.loads(run_card_path.read_text(encoding="utf-8"))
    run_card["variant"] = "token_opd"
    write_json(run_card_path, run_card)

    with pytest.raises(ValueError, match="expected block3_mean"):
        module.build_result(run_dir, historical)


def test_result_rejects_unverified_historical_grader_hash(tmp_path):
    module = load_module()
    run_dir, historical = make_fixture(tmp_path)
    hashes = (
        run_dir
        / "eval_step_100_n8"
        / "historical_external_grader"
        / "input_hashes.sha256"
    )
    hashes.write_text(
        f"{'0' * 64}  /grading/historical_utils_sha04f7.py\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="historical grader SHA"):
        module.build_result(run_dir, historical)


def test_result_rejects_grader_sha_attached_to_wrong_path(tmp_path):
    module = load_module()
    run_dir, historical = make_fixture(tmp_path)
    hashes = (
        run_dir
        / "eval_step_100_n8"
        / "historical_external_grader"
        / "input_hashes.sha256"
    )
    text = hashes.read_text(encoding="utf-8").replace(
        "historical_utils_sha04f7.py", "unrelated_grader.py"
    )
    hashes.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="historical grader SHA/path"):
        module.build_result(run_dir, historical)


def test_result_rejects_historical_eval_data_sha_mismatch(tmp_path):
    module = load_module()
    run_dir, historical = make_fixture(tmp_path)
    historical_data = json.loads(historical.read_text(encoding="utf-8"))
    historical_data["training"]["eval_sha256"] = "different-eval-data"
    write_json(historical, historical_data)

    with pytest.raises(ValueError, match="historical eval data SHA"):
        module.build_result(run_dir, historical)


def test_result_rejects_missing_eval_data_sha_on_both_sides(tmp_path):
    module = load_module()
    run_dir, historical = make_fixture(tmp_path)
    historical_data = json.loads(historical.read_text(encoding="utf-8"))
    historical_data["training"].pop("eval_sha256")
    write_json(historical, historical_data)
    manifest_path = run_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"].pop("test_parquet")
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="valid historical eval data SHA"):
        module.build_result(run_dir, historical)


def test_html_states_verdict_and_uses_relative_assets(tmp_path):
    module = load_module()
    run_dir, historical = make_fixture(tmp_path)
    result = module.build_result(run_dir, historical)
    report = tmp_path / "reports" / "report.html"
    assets = report.parent / "assets"
    report.parent.mkdir(parents=True)

    module.write_html(result, report, assets)
    text = report.read_text(encoding="utf-8")

    assert "未观察到持续性数值或固定评测崩塌" in text
    assert "Block3 相对 token OPD 的收益尚未被第二台机器独立验证" in text
    assert 'src="assets/position_heatmaps.png"' in text
    assert "没有同机 token baseline" in text
    assert text.count("<span>Diagnostics</span>") == 1
