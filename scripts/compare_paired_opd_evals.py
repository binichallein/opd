#!/usr/bin/env python3
"""Compare two OPD runs with exact rollout pairing and stratified bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TASKS = ("math500", "aime24", "aime25", "amc23")
TASK_COUNTS = {"math500": 500, "aime24": 30, "aime25": 30, "amc23": 83}
DEFAULT_STEPS = (50, 100, 200)
DIAGNOSTIC_STEPS = (1, *range(5, 201, 5))
HISTORICAL_GRADER_SHA256 = (
    "04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f"
)
TRAINING_CONTRACT_KEYS = (
    "source_commit",
    "revisiting_opd_base_commit",
    "student_model",
    "student_model_revision",
    "teacher_model",
    "teacher_model_revision",
    "train_data",
    "val_data",
    "expected_train_sha256",
    "seed",
    "data_seed",
    "rollout_seed",
    "environment_seed",
    "n_gpus_per_node",
    "train_batch_size",
    "ppo_mini_batch_size",
    "rollout_group_size",
    "max_prompt_length",
    "max_response_length",
    "learning_rate",
    "total_training_steps",
    "rollout_gpu_memory_utilization",
    "rollout_max_num_batched_tokens",
    "rollout_temperature",
    "rollout_top_p",
    "actor_ppo_micro_batch_size_per_gpu",
    "rollout_log_prob_micro_batch_size_per_gpu",
    "ref_log_prob_micro_batch_size_per_gpu",
    "opd_diagnostics",
    "opd_diag_interval",
    "opd_diag_topk",
    "opd_diag_position_bin",
    "opd_diag_position_stride",
    "opd_diag_sign_epsilon",
    "diagnostic_save_steps",
    "filter_overlong_prompts",
)
EXPECTED_TRAINING_VALUES = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-run", type=Path, required=True)
    parser.add_argument("--right-run", type=Path, required=True)
    parser.add_argument("--left-label", default="token_opd")
    parser.add_argument("--right-label", default="block3_mean")
    parser.add_argument(
        "--steps",
        default=",".join(str(step) for step in DEFAULT_STEPS),
    )
    parser.add_argument(
        "--view",
        choices=("outputs", "historical_external_grader"),
        default="historical_external_grader",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_712)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("steps must be positive")
    return steps


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing graded JSONL: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def load_sha256_manifest(path: Path) -> list[tuple[str, Path]]:
    if not path.is_file():
        raise ValueError(f"missing SHA-256 manifest: {path}")
    entries: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not _valid_sha256(fields[0]):
            raise ValueError(f"invalid SHA-256 manifest line in {path}: {line!r}")
        recorded_path = Path(fields[1].strip())
        if recorded_path in seen:
            raise ValueError(f"duplicate SHA-256 manifest path in {path}: {recorded_path}")
        seen.add(recorded_path)
        entries.append((fields[0].lower(), recorded_path))
    return entries


def _load_diagnostic_prompt_hashes(run_dir: Path) -> dict[int, str]:
    rows = load_jsonl(run_dir / "diagnostics" / "scalars.jsonl")
    result: dict[int, str] = {}
    for row in rows:
        step = int(row["step"])
        prompt_hash = row.get("prompt_batch_sha256")
        if step in result:
            raise ValueError(f"{run_dir}: duplicate diagnostic step {step}")
        if not _valid_sha256(prompt_hash):
            raise ValueError(f"{run_dir}: invalid prompt batch hash at step {step}")
        result[step] = str(prompt_hash).lower()
    if tuple(sorted(result)) != DIAGNOSTIC_STEPS:
        raise ValueError(f"{run_dir}: diagnostic prompt hash schedule is incomplete")
    return result


def _validate_data_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "data_manifest.json")
    counts = manifest.get("counts", {})
    expected_counts = {**TASK_COUNTS, "test_total": 643, "train": 1_791_700}
    for key, expected in expected_counts.items():
        if int(counts.get(key, -1)) != expected:
            raise ValueError(f"{run_dir}: data manifest count mismatch for {key}")
    hashes = manifest.get("sha256", {})
    required_hashes = (
        "train_parquet",
        "test_parquet",
        "math500_jsonl",
        "aime24_jsonl",
        "aime25_jsonl",
        "amc23_jsonl",
    )
    for key in required_hashes:
        if not _valid_sha256(hashes.get(key)):
            raise ValueError(f"{run_dir}: data manifest SHA is invalid for {key}")
    return {
        "counts": {key: int(counts[key]) for key in expected_counts},
        "sha256": {key: str(hashes[key]).lower() for key in required_hashes},
    }


def _validate_artifact_manifest(run_dir: Path, run_card: dict[str, Any]) -> set[tuple[str, str]]:
    entries = load_sha256_manifest(run_dir / "artifact_hashes.sha256")
    normalized = {(digest, str(path)) for digest, path in entries}
    if len(normalized) != len(entries):
        raise ValueError(f"{run_dir}: duplicate artifact hash entries")
    for model_key in ("student_model", "teacher_model"):
        model_root = str(run_card[model_key]).rstrip("/") + "/"
        model_entries = [
            (digest, path)
            for digest, path in normalized
            if path.startswith(model_root)
            and (path.endswith(".safetensors") or path.endswith(".safetensors.index.json"))
        ]
        if not model_entries:
            raise ValueError(f"{run_dir}: artifact manifest has no weights for {model_key}")
    return normalized


def validate_paired_training_contract(
    left_run: Path, right_run: Path
) -> dict[str, Any]:
    left_card = load_json(left_run / "run_card.json")
    right_card = load_json(right_run / "run_card.json")
    if left_card.get("variant") != "token_opd":
        raise ValueError(f"left variant must be token_opd, got {left_card.get('variant')!r}")
    if right_card.get("variant") != "block3_mean":
        raise ValueError(
            f"right variant must be block3_mean, got {right_card.get('variant')!r}"
        )
    for key in TRAINING_CONTRACT_KEYS:
        if left_card.get(key) != right_card.get(key):
            raise ValueError(f"paired training contract differs at {key}")
    for key, expected in EXPECTED_TRAINING_VALUES.items():
        if left_card.get(key) != expected:
            raise ValueError(
                f"paired training contract {key} expected {expected!r}, "
                f"got {left_card.get(key)!r}"
            )

    left_data = _validate_data_manifest(left_run)
    right_data = _validate_data_manifest(right_run)
    if left_data != right_data:
        raise ValueError("paired data manifests differ")
    if left_card.get("expected_train_sha256") != left_data["sha256"]["train_parquet"]:
        raise ValueError("left run-card train SHA differs from data manifest")

    left_artifacts = _validate_artifact_manifest(left_run, left_card)
    right_artifacts = _validate_artifact_manifest(right_run, right_card)
    if left_artifacts != right_artifacts:
        raise ValueError("paired artifact manifests differ")

    left_prompts = _load_diagnostic_prompt_hashes(left_run)
    right_prompts = _load_diagnostic_prompt_hashes(right_run)
    if left_prompts != right_prompts:
        differing = [
            step for step in DIAGNOSTIC_STEPS if left_prompts[step] != right_prompts[step]
        ]
        raise ValueError(f"paired prompt batch hashes differ at steps {differing}")
    return {
        "left_variant": left_card["variant"],
        "right_variant": right_card["variant"],
        "source_commit": left_card["source_commit"],
        "student_model": left_card["student_model"],
        "student_model_revision": left_card.get("student_model_revision"),
        "teacher_model": left_card["teacher_model"],
        "teacher_model_revision": left_card.get("teacher_model_revision"),
        "train_sha256": left_data["sha256"]["train_parquet"],
        "test_sha256": left_data["sha256"]["test_parquet"],
        "prompt_batch_hashes_matched": len(left_prompts),
    }


def _view_dir(run_dir: Path, step: int, view: str) -> Path:
    return run_dir / f"eval_step_{step}_n8" / view


def _expected_grader(view: str) -> str:
    return "verl" if view == "outputs" else "external"


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def validate_run_acceptance(run_dir: Path, steps: tuple[int, ...]) -> None:
    acceptance = load_json(run_dir / "acceptance.json")
    if acceptance.get("passed") is not True or acceptance.get("issues"):
        raise ValueError(
            f"{run_dir}: final acceptance failed: {acceptance.get('issues', [])}"
        )
    for key in ("checkpoint_steps", "eval_steps"):
        observed = {int(step) for step in acceptance.get(key, [])}
        if not set(steps).issubset(observed):
            raise ValueError(f"{run_dir}: acceptance {key} does not cover {steps}")
    exit_path = run_dir / "exit_code.txt"
    if not exit_path.is_file() or exit_path.read_text(encoding="utf-8").strip() != "0":
        raise ValueError(f"{run_dir}: training exit code is missing or nonzero")


def validate_eval_config(run_dir: Path, step: int) -> dict[str, Any]:
    config = load_json(
        run_dir / f"eval_step_{step}_n8" / "outputs" / "eval_config.json"
    )
    expected = {
        "n": 8,
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 16384,
        "eval_seed": 21,
        "rollout_seeds": list(range(21, 29)),
        "grader": "verl",
        "enable_thinking": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"step {step} {run_dir}: eval config {key} expected {value!r}, "
                f"got {config.get(key)!r}"
            )
    if set(config.get("tasks", [])) != set(TASKS):
        raise ValueError(f"step {step} {run_dir}: eval config task set is incomplete")
    return config


def validate_external_manifest(
    run_dir: Path, step: int
) -> dict[str, dict[tuple[str, str, int, int], dict[str, Any]]]:
    view = run_dir / f"eval_step_{step}_n8" / "historical_external_grader"
    input_entries = load_sha256_manifest(view / "input_hashes.sha256")
    grader_entries = [
        (digest, path)
        for digest, path in input_entries
        if str(path).endswith("/grading/historical_utils_sha04f7.py")
    ]
    if len(grader_entries) != 1 or grader_entries[0][0] != HISTORICAL_GRADER_SHA256:
        raise ValueError(f"step {step} {run_dir}: historical grader SHA is unverified")
    grader_digest, grader_path = grader_entries[0]
    if not grader_path.is_file() or hashlib.sha256(grader_path.read_bytes()).hexdigest() != grader_digest:
        raise ValueError(f"step {step} {run_dir}: historical grader SHA mismatch")

    expected_raw = {
        task: run_dir
        / f"eval_step_{step}_n8"
        / "outputs"
        / f"{task}_t1.0_p0.9_n8-MNT16384.jsonl"
        for task in TASKS
    }
    expected_primary = {
        task: run_dir
        / f"eval_step_{step}_n8"
        / "outputs"
        / f"{task}_graded.jsonl"
        for task in TASKS
    }
    expected_metadata = (
        run_dir / f"eval_step_{step}_n8" / "outputs" / "eval_config.json",
        run_dir / f"eval_step_{step}_n8" / "outputs" / "summary.json",
        run_dir / "acceptance.json",
    )
    if len(input_entries) != 12:
        raise ValueError(f"step {step} {run_dir}: external input hash set is incomplete")
    input_by_path = {str(path): digest for digest, path in input_entries}

    def verified_input_path(expected_path: Path, label: str) -> Path:
        matching = [
            (Path(path_string), digest)
            for path_string, digest in input_by_path.items()
            if path_string.endswith(str(expected_path))
        ]
        if len(matching) != 1:
            raise ValueError(f"step {step} {run_dir}: {label} input path is invalid")
        path, digest = matching[0]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"step {step} {run_dir}: {label} input SHA mismatch: {path}")
        return path

    for metadata_path in expected_metadata:
        verified_input_path(metadata_path, "accepted metadata")
    raw_rows: dict[str, dict[tuple[str, str, int, int], dict[str, Any]]] = {}
    for task, expected_path in expected_raw.items():
        path = verified_input_path(expected_path, f"raw-output {task}")
        task_rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        for row in load_jsonl(path):
            key = (
                str(row.get("task")),
                str(row["example_id"]),
                int(row["rollout_id"]),
                int(row["seed"]),
            )
            if key in task_rows:
                raise ValueError(f"step {step} {run_dir}: duplicate raw rollout key {key}")
            task_rows[key] = row
        raw_rows[task] = task_rows

    for task, expected_path in expected_primary.items():
        path = verified_input_path(expected_path, f"primary graded {task}")
        primary_rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        for row in load_jsonl(path):
            key = (
                str(row.get("task")),
                str(row["example_id"]),
                int(row["rollout_id"]),
                int(row["seed"]),
            )
            if type(row.get("correct")) is not bool:
                raise ValueError(
                    f"step {step} {run_dir}: primary correct must be a JSON boolean"
                )
            if key in primary_rows:
                raise ValueError(f"step {step} {run_dir}: duplicate primary key {key}")
            primary_rows[key] = row
        if set(primary_rows) != set(raw_rows[task]):
            raise ValueError(
                f"step {step} {run_dir}: raw output differs from primary graded evidence"
            )
        for key, raw in raw_rows[task].items():
            primary_without_score = {
                name: value for name, value in primary_rows[key].items() if name != "correct"
            }
            if primary_without_score != raw:
                raise ValueError(
                    f"step {step} {run_dir}: raw output differs from primary graded evidence"
                )

    output_entries = load_sha256_manifest(view / "output_hashes.sha256")
    expected_outputs = {
        *(view / f"{task}_graded.jsonl" for task in TASKS),
        view / "summary.json",
    }
    if len(output_entries) != 5:
        raise ValueError(f"step {step} {run_dir}: external output hash set is incomplete")
    matched_outputs: set[Path] = set()
    for digest, recorded_path in output_entries:
        candidates = [path for path in expected_outputs if str(recorded_path).endswith(str(path))]
        if len(candidates) != 1:
            raise ValueError(f"step {step} {run_dir}: external output hash path is invalid")
        path = recorded_path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"step {step} {run_dir}: external output SHA mismatch: {path}")
        matched_outputs.add(candidates[0])
    if matched_outputs != expected_outputs:
        raise ValueError(f"step {step} {run_dir}: external output hash set is incomplete")
    return raw_rows


def load_step_view(
    run_dir: Path,
    step: int,
    view: str,
) -> dict[str, Any]:
    root = _view_dir(run_dir, step, view)
    config = validate_eval_config(run_dir, step)
    raw_rows = None
    if view == "historical_external_grader":
        raw_rows = validate_external_manifest(run_dir, step)
    summary = load_json(root / "summary.json")
    expected_grader = _expected_grader(view)
    if summary.get("grader") != expected_grader:
        raise ValueError(
            f"step {step} {run_dir}: grader expected {expected_grader!r}, "
            f"got {summary.get('grader')!r}"
        )
    if int(summary.get("n_expected", -1)) != 8:
        raise ValueError(f"step {step} {run_dir}: expected n=8")
    expected_seeds = tuple(range(21, 29))
    if tuple(summary.get("rollout_seeds", [])) != expected_seeds:
        raise ValueError(f"step {step} {run_dir}: expected rollout seeds 21-28")
    if int(summary.get("eval_seed", -1)) != 21:
        raise ValueError(f"step {step} {run_dir}: expected eval seed 21")
    if summary.get("enable_thinking") is not False:
        raise ValueError(f"step {step} {run_dir}: enable_thinking must be false")
    if set(summary.get("tasks", {})) != set(TASKS):
        raise ValueError(f"step {step} {run_dir}: task set is incomplete")

    task_data: dict[str, Any] = {}
    all_keys: set[tuple[str, str, int, int]] = set()
    for task in TASKS:
        rows = load_jsonl(root / f"{task}_graded.jsonl")
        groups: dict[str, list[dict[str, Any]]] = {}
        keys: set[tuple[str, str, int, int]] = set()
        for row in rows:
            row_task = str(row.get("task"))
            if row_task != task:
                raise ValueError(f"step {step} {run_dir}: row task mismatch for {task}")
            example_id = str(row["example_id"])
            rollout_id = int(row["rollout_id"])
            seed = int(row["seed"])
            key = (task, example_id, rollout_id, seed)
            if key in keys:
                raise ValueError(f"step {step} {run_dir}: duplicate paired rollout key {key}")
            if type(row.get("correct")) is not bool:
                raise ValueError(
                    f"step {step} {run_dir}: correct must be a JSON boolean for {key}"
                )
            if raw_rows is not None:
                raw = raw_rows[task].get(key)
                if raw is None:
                    raise ValueError(f"step {step} {run_dir}: graded/raw key mismatch {key}")
                without_score = {name: value for name, value in row.items() if name != "correct"}
                if without_score != raw:
                    raise ValueError(
                        f"step {step} {run_dir}: external grader changed raw fields for {key}"
                    )
            keys.add(key)
            all_keys.add(key)
            groups.setdefault(example_id, []).append(row)

        prompt_metrics: dict[str, dict[str, float]] = {}
        for example_id, prompt_rows in groups.items():
            observed_seeds = {int(row["seed"]) for row in prompt_rows}
            observed_rollouts = {int(row["rollout_id"]) for row in prompt_rows}
            if (
                len(prompt_rows) != 8
                or observed_seeds != set(expected_seeds)
                or observed_rollouts != set(range(8))
            ):
                raise ValueError(
                    f"step {step} {run_dir} {task}/{example_id}: "
                    "expected exactly 8 rollouts with rollout_id 0-7 and seeds 21-28"
                )
            scores = [float(row["correct"]) for row in prompt_rows]
            prompt_metrics[example_id] = {
                "avg_at_n": sum(scores) / 8,
                "pass_at_n": float(any(scores)),
            }

        task_summary = summary["tasks"][task]
        if len(groups) != TASK_COUNTS[task]:
            raise ValueError(
                f"step {step} {run_dir} {task}: expected {TASK_COUNTS[task]} prompts, "
                f"got {len(groups)}"
            )
        if raw_rows is not None and set(raw_rows[task]) != keys:
            raise ValueError(f"step {step} {run_dir} {task}: graded/raw key sets differ")
        if int(task_summary.get("num_examples", -1)) != len(groups):
            raise ValueError(f"step {step} {run_dir} {task}: example count mismatch")
        if int(task_summary.get("total_rollouts", -1)) != len(rows):
            raise ValueError(f"step {step} {run_dir} {task}: rollout count mismatch")
        recomputed_avg = sum(v["avg_at_n"] for v in prompt_metrics.values()) / len(groups)
        recomputed_pass = sum(v["pass_at_n"] for v in prompt_metrics.values()) / len(groups)
        if not _close(recomputed_avg, float(task_summary["avg_at_n"])):
            raise ValueError(f"step {step} {run_dir} {task}: avg_at_n mismatch")
        if not _close(recomputed_pass, float(task_summary["pass_at_n"])):
            raise ValueError(f"step {step} {run_dir} {task}: pass_at_n mismatch")
        task_data[task] = {
            "keys": keys,
            "prompt_metrics": prompt_metrics,
            "avg_at_n": recomputed_avg,
            "pass_at_n": recomputed_pass,
        }

    macro_avg = sum(task_data[task]["avg_at_n"] for task in TASKS) / len(TASKS)
    macro_pass = sum(task_data[task]["pass_at_n"] for task in TASKS) / len(TASKS)
    if not _close(macro_avg, float(summary["macro_avg_at_n"])):
        raise ValueError(f"step {step} {run_dir}: macro_avg_at_n mismatch")
    if not _close(macro_pass, float(summary["macro_pass_at_n"])):
        raise ValueError(f"step {step} {run_dir}: macro_pass_at_n mismatch")
    return {
        "summary": summary,
        "config": config,
        "tasks": task_data,
        "keys": all_keys,
        "macro_avg_at_n": macro_avg,
        "macro_pass_at_n": macro_pass,
    }


def _outcomes(values: list[float], epsilon: float = 1e-12) -> dict[str, int]:
    return {
        "win": sum(value > epsilon for value in values),
        "tie": sum(abs(value) <= epsilon for value in values),
        "loss": sum(value < -epsilon for value in values),
    }


def stratified_bootstrap(
    differences: dict[str, dict[str, np.ndarray]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    rng = np.random.default_rng(seed)
    samples = {
        "avg_at_n": np.empty(replicates, dtype=np.float64),
        "pass_at_n": np.empty(replicates, dtype=np.float64),
    }
    for replicate in range(replicates):
        macro = {metric: [] for metric in samples}
        for task in TASKS:
            count = len(differences[task]["avg_at_n"])
            indices = rng.integers(0, count, size=count)
            for metric in samples:
                macro[metric].append(float(differences[task][metric][indices].mean()))
        for metric in samples:
            samples[metric][replicate] = float(np.mean(macro[metric]))
    result = {}
    for metric, values in samples.items():
        lower, upper = np.percentile(values, [2.5, 97.5])
        result[metric] = {
            "lower_95": float(lower),
            "upper_95": float(upper),
            "bootstrap_mean": float(values.mean()),
        }
    return result


def compare_step(
    left_run: Path,
    right_run: Path,
    step: int,
    view: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    left = load_step_view(left_run, step, view)
    right = load_step_view(right_run, step, view)
    contract_keys = (
        "tasks",
        "n",
        "temperature",
        "top_p",
        "max_tokens",
        "eval_seed",
        "rollout_seeds",
        "grader",
        "enable_thinking",
    )
    for key in contract_keys:
        if left["config"].get(key) != right["config"].get(key):
            raise ValueError(f"step {step}: paired eval config differs for {key}")
    if left["keys"] != right["keys"]:
        missing_right = len(left["keys"] - right["keys"])
        missing_left = len(right["keys"] - left["keys"])
        raise ValueError(
            f"step {step}: paired rollout keys differ; "
            f"missing_right={missing_right}, missing_left={missing_left}"
        )

    differences: dict[str, dict[str, np.ndarray]] = {}
    per_task = {}
    pooled = {"avg_at_n": [], "pass_at_n": []}
    for task in TASKS:
        left_prompts = left["tasks"][task]["prompt_metrics"]
        right_prompts = right["tasks"][task]["prompt_metrics"]
        if set(left_prompts) != set(right_prompts):
            raise ValueError(f"step {step} {task}: paired prompt ids differ")
        prompt_ids = sorted(left_prompts)
        task_differences = {}
        for metric in ("avg_at_n", "pass_at_n"):
            values = np.asarray(
                [
                    right_prompts[prompt_id][metric] - left_prompts[prompt_id][metric]
                    for prompt_id in prompt_ids
                ],
                dtype=np.float64,
            )
            task_differences[metric] = values
            pooled[metric].extend(float(value) for value in values)
        differences[task] = task_differences
        per_task[task] = {
            metric: {
                "left": float(left["tasks"][task][metric]),
                "right": float(right["tasks"][task][metric]),
                "delta": float(task_differences[metric].mean()),
            }
            for metric in ("avg_at_n", "pass_at_n")
        }

    macro = {}
    for metric, left_key in (
        ("avg_at_n", "macro_avg_at_n"),
        ("pass_at_n", "macro_pass_at_n"),
    ):
        left_value = float(left[left_key])
        right_value = float(right[left_key])
        macro[metric] = {
            "left": left_value,
            "right": right_value,
            "delta": right_value - left_value,
        }
    bootstrap = stratified_bootstrap(
        differences,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "step": step,
        "pair_count": sum(len(differences[task]["avg_at_n"]) for task in TASKS),
        "rollout_key_count": len(left["keys"]),
        "per_task": per_task,
        "macro": macro,
        "prompt_outcomes": {
            metric: _outcomes(values) for metric, values in pooled.items()
        },
        "bootstrap": bootstrap,
    }


def compare_runs(
    left_run: Path,
    right_run: Path,
    *,
    steps: tuple[int, ...] = DEFAULT_STEPS,
    view: str = "historical_external_grader",
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_260_712,
    left_label: str = "token_opd",
    right_label: str = "block3_mean",
) -> dict[str, Any]:
    if view not in {"outputs", "historical_external_grader"}:
        raise ValueError(f"unsupported view: {view}")
    left_run = left_run.resolve(strict=True)
    right_run = right_run.resolve(strict=True)
    training_contract = validate_paired_training_contract(left_run, right_run)
    validate_run_acceptance(left_run, steps)
    validate_run_acceptance(right_run, steps)
    comparisons = {
        str(step): compare_step(
            left_run,
            right_run,
            step,
            view,
            bootstrap_replicates,
            bootstrap_seed,
        )
        for step in steps
    }
    final = comparisons[str(max(steps))]
    avg_delta = float(final["macro"]["avg_at_n"]["delta"])
    pass_delta = float(final["macro"]["pass_at_n"]["delta"])
    avg_lower = float(final["bootstrap"]["avg_at_n"]["lower_95"])
    pass_lower = float(final["bootstrap"]["pass_at_n"]["lower_95"])
    if avg_delta <= 0 or pass_delta <= 0:
        status = "not_supported"
    elif avg_lower > 0 and pass_lower > 0:
        status = "strong_single_seed_support"
    else:
        status = "directional_support"
    return {
        "left": {"label": left_label, "run_dir": str(left_run)},
        "right": {"label": right_label, "run_dir": str(right_run)},
        "training_contract": training_contract,
        "grader_view": view,
        "bootstrap": {
            "method": "task-stratified paired prompt bootstrap",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "steps": comparisons,
        "decision": {
            "step": max(steps),
            "status": status,
            "scope": "one paired training seed; bootstrap does not estimate training-run variance",
        },
    }


def main() -> None:
    args = parse_args()
    result = compare_runs(
        args.left_run,
        args.right_run,
        steps=parse_steps(args.steps),
        view=args.view,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        left_label=args.left_label,
        right_label=args.right_label,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
