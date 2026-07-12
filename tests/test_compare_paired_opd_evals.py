import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_paired_opd_evals.py"
TASKS = ("math500", "aime24", "aime25", "amc23")
TASK_COUNTS = {"math500": 500, "aime24": 30, "aime25": 30, "amc23": 83}
FIXTURE_GRADER = b"# fixture grader\n"
FIXTURE_GRADER_SHA256 = hashlib.sha256(FIXTURE_GRADER).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("paired_opd_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.HISTORICAL_GRADER_SHA256 = FIXTURE_GRADER_SHA256
    return module


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_run(
    run_dir: Path,
    *,
    right: bool,
    drop_last: bool = False,
    view_name: str = "historical_external_grader",
):
    view = run_dir / "eval_step_200_n8" / view_name
    outputs = run_dir / "eval_step_200_n8" / "outputs"
    write_json(
        run_dir / "acceptance.json",
        {
            "passed": True,
            "issues": [],
            "checkpoint_steps": [50, 100, 200],
            "diagnostic_steps": [1, *range(5, 201, 5)],
            "eval_steps": [50, 100, 200],
        },
    )
    (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    write_json(
        run_dir / "run_card.json",
        {
            "variant": "block3_mean" if right else "token_opd",
            "source_commit": "source-commit",
            "revisiting_opd_base_commit": "upstream-commit",
            "student_model": "/models/student",
            "student_model_revision": "student-revision",
            "teacher_model": "/models/teacher",
            "teacher_model_revision": "teacher-revision",
            "train_data": "/data/train.parquet",
            "val_data": "/data/test.parquet",
            "expected_train_sha256": "a" * 64,
            "seed": 21,
            "data_seed": 21,
            "rollout_seed": 21,
            "environment_seed": 21,
            "n_gpus_per_node": 4,
            "train_batch_size": 4,
            "ppo_mini_batch_size": 32,
            "rollout_group_size": 8,
            "max_prompt_length": 2048,
            "max_response_length": 16384,
            "learning_rate": 2e-6,
            "total_training_steps": 200,
            "rollout_gpu_memory_utilization": 0.6,
            "rollout_max_num_batched_tokens": 18432,
            "rollout_temperature": 1.0,
            "rollout_top_p": 0.9,
            "actor_ppo_micro_batch_size_per_gpu": 1,
            "rollout_log_prob_micro_batch_size_per_gpu": 4,
            "ref_log_prob_micro_batch_size_per_gpu": 1,
            "opd_diagnostics": True,
            "opd_diag_interval": 5,
            "opd_diag_topk": 16,
            "opd_diag_position_bin": 128,
            "opd_diag_position_stride": 1,
            "opd_diag_sign_epsilon": 1e-4,
            "diagnostic_save_steps": "50,100,200",
            "filter_overlong_prompts": False,
        },
    )
    write_json(
        run_dir / "data_manifest.json",
        {
            "counts": {**TASK_COUNTS, "test_total": 643, "train": 1_791_700},
            "sha256": {
                "train_parquet": "a" * 64,
                "test_parquet": "b" * 64,
                "math500_jsonl": "c" * 64,
                "aime24_jsonl": "d" * 64,
                "aime25_jsonl": "e" * 64,
                "amc23_jsonl": "f" * 64,
            },
        },
    )
    (run_dir / "artifact_hashes.sha256").write_text(
        f"{'a' * 64}  /data/train.parquet\n"
        f"{'b' * 64}  /data/test.parquet\n"
        f"{'1' * 64}  /models/student/model.safetensors\n"
        f"{'2' * 64}  /models/teacher/model.safetensors\n",
        encoding="utf-8",
    )
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True)
    diagnostic_rows = [
        {"step": step, "prompt_batch_sha256": f"{step:064x}"}
        for step in (1, *range(5, 201, 5))
    ]
    (diagnostics / "scalars.jsonl").write_text(
        "\n".join(json.dumps(row) for row in diagnostic_rows) + "\n",
        encoding="utf-8",
    )
    write_json(
        outputs / "eval_config.json",
        {
            "tasks": list(TASKS),
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
    summary = {
        "n_expected": 8,
        "grader": "external",
        "eval_seed": 21,
        "rollout_seeds": list(range(21, 29)),
        "enable_thinking": False,
        "tasks": {},
        "macro_avg_at_n": 1.0 if right else 0.0,
        "macro_pass_at_n": 1.0 if right else 0.0,
    }
    for task in TASKS:
        rows = []
        for example_index in range(TASK_COUNTS[task]):
            for rollout_id in range(8):
                rows.append(
                    {
                        "task": task,
                        "example_id": f"{task}-{example_index}",
                        "rollout_id": rollout_id,
                        "seed": 21 + rollout_id,
                        "correct": bool(right),
                    }
                )
        if drop_last and task == "math500":
            rows.pop()
        (view / f"{task}_graded.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (view / f"{task}_graded.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        (outputs / f"{task}_graded.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        raw_path = outputs / f"{task}_t1.0_p0.9_n8-MNT16384.jsonl"
        raw_path.write_text(
            "\n".join(
                json.dumps({key: value for key, value in row.items() if key != "correct"})
                for row in rows
            )
            + "\n"
        )
        summary["tasks"][task] = {
            "num_examples": TASK_COUNTS[task],
            "total_rollouts": len(rows),
            "avg_at_n": 1.0 if right else 0.0,
            "pass_at_n": 1.0 if right else 0.0,
        }
    write_json(view / "summary.json", summary)
    write_json(outputs / "summary.json", {**summary, "grader": "verl"})
    grader = run_dir / "grading" / "historical_utils_sha04f7.py"
    grader.parent.mkdir(parents=True, exist_ok=True)
    grader.write_bytes(FIXTURE_GRADER)
    manifest = []
    for task in TASKS:
        raw_path = outputs / f"{task}_t1.0_p0.9_n8-MNT16384.jsonl"
        manifest.append(
            f"{hashlib.sha256(raw_path.read_bytes()).hexdigest()}  {raw_path}"
        )
    for task in TASKS:
        primary_graded = outputs / f"{task}_graded.jsonl"
        manifest.append(
            f"{hashlib.sha256(primary_graded.read_bytes()).hexdigest()}  {primary_graded}"
        )
    for path in (
        outputs / "eval_config.json",
        outputs / "summary.json",
        run_dir / "acceptance.json",
    ):
        manifest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")
    manifest.append(
        f"{FIXTURE_GRADER_SHA256}  "
        f"{grader}"
    )
    (view / "input_hashes.sha256").write_text("\n".join(manifest) + "\n")
    output_paths = [view / f"{task}_graded.jsonl" for task in TASKS]
    output_paths.append(view / "summary.json")
    (view / "output_hashes.sha256").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}"
            for path in output_paths
        )
        + "\n",
        encoding="utf-8",
    )


def test_exact_paired_comparison_reports_macro_and_prompt_wins(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)

    result = module.compare_runs(
        left,
        right,
        steps=(200,),
        view="historical_external_grader",
        bootstrap_replicates=200,
        bootstrap_seed=20260712,
    )

    step = result["steps"]["200"]
    assert step["macro"]["avg_at_n"]["delta"] == pytest.approx(1.0)
    assert step["macro"]["pass_at_n"]["delta"] == pytest.approx(1.0)
    assert step["prompt_outcomes"]["avg_at_n"] == {
        "win": 643,
        "tie": 0,
        "loss": 0,
    }
    assert step["pair_count"] == 643
    assert result["decision"]["status"] == "strong_single_seed_support"


def test_comparison_rejects_missing_paired_rollout_key(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True, drop_last=True)

    with pytest.raises(ValueError, match="exactly 8 rollouts|paired rollout keys"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_bootstrap_is_deterministic_for_fixed_seed(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)

    kwargs = {
        "steps": (200,),
        "view": "historical_external_grader",
        "bootstrap_replicates": 100,
        "bootstrap_seed": 17,
    }
    first = module.compare_runs(left, right, **kwargs)
    second = module.compare_runs(left, right, **kwargs)

    assert first["steps"]["200"]["bootstrap"] == second["steps"]["200"]["bootstrap"]
    source = SCRIPT.read_text(encoding="utf-8")
    assert "seed=bootstrap_seed + step" not in source
    assert "seed=bootstrap_seed," in source


def test_comparison_rejects_wrong_grader_view_contract(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    summary_path = (
        right
        / "eval_step_200_n8"
        / "historical_external_grader"
        / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["grader"] = "verl"
    write_json(summary_path, summary)

    with pytest.raises(ValueError, match="grader"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_rejects_eval_sampling_contract_mismatch(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    config_path = right / "eval_step_200_n8" / "outputs" / "eval_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["temperature"] = 0.7
    write_json(config_path, config)

    with pytest.raises(ValueError, match="temperature"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_rejects_unverified_historical_grader(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    manifest = (
        right
        / "eval_step_200_n8"
        / "historical_external_grader"
        / "input_hashes.sha256"
    )
    manifest.write_text(
        manifest.read_text().replace(FIXTURE_GRADER_SHA256[:8], "deadbeef")
    )

    with pytest.raises(ValueError, match="historical grader SHA"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_rejects_failed_final_acceptance(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    acceptance_path = right / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["passed"] = False
    acceptance["issues"] = ["eval missing"]
    write_json(acceptance_path, acceptance)

    with pytest.raises(ValueError, match="acceptance"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_rejects_mismatched_training_contract(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    run_card_path = right / "run_card.json"
    run_card = json.loads(run_card_path.read_text(encoding="utf-8"))
    run_card["teacher_model_revision"] = "different-teacher"
    write_json(run_card_path, run_card)

    with pytest.raises(ValueError, match="training contract.*teacher_model_revision"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_rejects_wrong_left_or_right_variant(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    run_card_path = right / "run_card.json"
    run_card = json.loads(run_card_path.read_text(encoding="utf-8"))
    run_card["variant"] = "token_opd"
    write_json(run_card_path, run_card)

    with pytest.raises(ValueError, match="right variant"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_rejects_mismatched_data_manifest(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    manifest_path = right / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"]["test_parquet"] = "0" * 64
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="data manifest"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_rejects_different_prompt_batches(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    diagnostics_path = right / "diagnostics" / "scalars.jsonl"
    rows = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]
    rows[5]["prompt_batch_sha256"] = "f" * 64
    diagnostics_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="prompt batch"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_rejects_tampered_grader_file(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    (right / "grading" / "historical_utils_sha04f7.py").write_text(
        "# tampered\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="grader SHA"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_rejects_tampered_external_graded_output(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    graded = (
        right
        / "eval_step_200_n8"
        / "historical_external_grader"
        / "math500_graded.jsonl"
    )
    graded.write_text(graded.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="external output SHA"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_requires_correct_to_be_json_boolean(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    view = right / "eval_step_200_n8" / "historical_external_grader"
    graded = view / "math500_graded.jsonl"
    rows = graded.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["correct"] = "false"
    rows[0] = json.dumps(first)
    graded.write_text("\n".join(rows) + "\n", encoding="utf-8")
    output_paths = [view / f"{task}_graded.jsonl" for task in TASKS]
    output_paths.append(view / "summary.json")
    (view / "output_hashes.sha256").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}"
            for path in output_paths
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="correct must be a JSON boolean"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_rejects_raw_that_no_longer_matches_primary_graded_evidence(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    write_run(left, right=False)
    write_run(right, right=True)
    raw_path = (
        right
        / "eval_step_200_n8"
        / "outputs"
        / "math500_t1.0_p0.9_n8-MNT16384.jsonl"
    )
    rows = raw_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["response"] = "tampered-after-acceptance"
    rows[0] = json.dumps(first)
    raw_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = (
        right
        / "eval_step_200_n8"
        / "historical_external_grader"
        / "input_hashes.sha256"
    )
    manifest_lines = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        _, recorded_path = line.split(maxsplit=1)
        if Path(recorded_path) == raw_path:
            line = f"{hashlib.sha256(raw_path.read_bytes()).hexdigest()}  {raw_path}"
        manifest_lines.append(line)
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="raw output differs from primary graded evidence"):
        module.compare_runs(
            left,
            right,
            steps=(200,),
            view="historical_external_grader",
            bootstrap_replicates=10,
        )


def test_comparison_supports_non_destructive_audited_external_view(tmp_path):
    module = load_module()
    left = tmp_path / "token"
    right = tmp_path / "block3"
    view = "historical_external_grader_audited"
    write_run(left, right=False, view_name=view)
    write_run(right, right=True, view_name=view)

    result = module.compare_runs(
        left,
        right,
        steps=(200,),
        view=view,
        bootstrap_replicates=10,
    )

    assert result["grader_view"] == view
    assert result["decision"]["status"] == "strong_single_seed_support"
