#!/usr/bin/env python3
"""Audit one completed OPD diagnostic run before analysis."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re


EXPECTED_DIAGNOSTIC_STEPS = [1, *range(5, 201, 5)]
EXPECTED_CHECKPOINT_STEPS = [40, 50, 60, 80, 100, 200]
EXPECTED_EVAL_STEPS = [50, 100, 200]
EXPECTED_TASKS = {"math500", "aime24", "aime25", "amc23"}
EXPECTED_TASK_EXAMPLES = {
    "math500": 500,
    "aime24": 30,
    "aime25": 30,
    "amc23": 83,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", default="block10_mean")
    parser.add_argument(
        "--checkpoint-steps",
        default=",".join(str(step) for step in EXPECTED_CHECKPOINT_STEPS),
    )
    parser.add_argument(
        "--eval-steps",
        default=",".join(str(step) for step in EXPECTED_EVAL_STEPS),
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--expected-train-sha256")
    parser.add_argument("--expected-student-model-suffix")
    parser.add_argument("--expected-teacher-model-suffix")
    parser.add_argument("--expected-total-training-steps", type=int, default=200)
    parser.add_argument("--expected-diagnostic-steps")
    parser.add_argument("--expected-diag-interval", type=int, default=5)
    parser.add_argument("--expected-resume-mode", default="disable")
    parser.add_argument("--expected-resume-from-path")
    parser.add_argument("--expected-source-commit")
    return parser.parse_args()


def parse_step_list(value: str) -> list[int]:
    """Parse a comma-separated milestone list into sorted unique positive steps."""
    if not value.strip():
        return []
    steps = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if any(step <= 0 for step in steps):
        raise ValueError("milestone steps must be positive")
    return steps


def expected_run_card(
    variant: str,
    total_training_steps: int = 200,
    diag_interval: int = 5,
    resume_mode: str = "disable",
) -> dict[str, object]:
    """Return the invariant run-card fields required for diagnostic experiments."""
    return {
        "variant": variant,
        "seed": 21,
        "data_seed": 21,
        "rollout_seed": 21,
        "environment_seed": 21,
        "train_batch_size": 4,
        "ppo_mini_batch_size": 32,
        "rollout_group_size": 8,
        "max_prompt_length": 2048,
        "max_response_length": 16384,
        "learning_rate": 2e-6,
        "total_training_steps": total_training_steps,
        "test_freq": -1,
        "opd_diagnostics": True,
        "opd_diag_interval": diag_interval,
        "opd_diag_topk": 16,
        "opd_diag_position_stride": 1,
        "rollout_gpu_memory_utilization": 0.6,
        "actor_ppo_micro_batch_size_per_gpu": 1,
        "rollout_log_prob_micro_batch_size_per_gpu": 4,
        "ref_log_prob_micro_batch_size_per_gpu": 1,
        "rollout_temperature": 1.0,
        "rollout_top_p": 0.9,
        "resume_mode": resume_mode,
        "filter_overlong_prompts": False,
    }


def checkpoint_issues(run_dir: Path, step: int, world_size: int = 4) -> list[str]:
    root = run_dir / "checkpoints" / f"global_step_{step}"
    actor = root / "actor"
    issues = []
    if not (root / "data.pt").is_file():
        issues.append(f"step {step}: missing data.pt")
    for prefix in ("model", "optim", "extra_state"):
        files = list(actor.glob(f"{prefix}_world_size_{world_size}_rank_*.pt"))
        rank_pattern = re.compile(
            rf"^{re.escape(prefix)}_world_size_{world_size}_rank_(\d+)\.pt$"
        )
        ranks = sorted(
            int(match.group(1))
            for path in files
            if (match := rank_pattern.match(path.name)) is not None
        )
        if (
            len(files) != world_size
            or ranks != list(range(world_size))
            or any(path.stat().st_size == 0 for path in files)
        ):
            issues.append(
                f"step {step}: expected nonempty {prefix} ranks "
                f"{list(range(world_size))}, got ranks {ranks}"
            )
    return issues


def numerical_scalar_issues(records: list[dict[str, object]]) -> list[str]:
    """Reject non-finite metrics and nonzero numerical-failure counters."""
    issues = []
    error_count_suffixes = (
        "_nonfinite_count",
        "_overflow_count",
        "_underflow_count",
    )
    for record in records:
        step = int(record["step"])
        for key, value in record.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                issues.append(f"step {step}: nonfinite scalar {key}={value}")
            elif key.endswith(error_count_suffixes) and numeric_value > 0:
                issues.append(f"step {step}: numerical error counter {key}={value}")
    return issues


def artifact_hash_issues(
    manifest_path: Path, expected_train_sha256: str | None
) -> list[str]:
    if expected_train_sha256 is None:
        return []
    if not manifest_path.is_file():
        return ["cannot verify train.parquet SHA-256: artifact manifest is missing"]
    train_entries = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2 and Path(fields[1].strip()).name == "train.parquet":
            train_entries.append(fields[0])
    if train_entries != [expected_train_sha256]:
        return [
            "train.parquet SHA-256 mismatch: expected "
            f"{expected_train_sha256}, got {train_entries}"
        ]
    return []


def eval_issues(run_dir: Path, step: int) -> list[str]:
    eval_dir = run_dir / f"eval_step_{step}_n8"
    summary_path = eval_dir / "outputs" / "summary.json"
    if not summary_path.is_file():
        return [f"step {step}: missing n=8 eval summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = summary.get("tasks", {})
    issues = []
    expected_eval = {
        "n_expected": 8,
        "grader": "verl",
        "eval_seed": 21,
        "rollout_seeds": list(range(21, 29)),
        "enable_thinking": False,
    }
    for key, expected in expected_eval.items():
        if summary.get(key) != expected:
            issues.append(
                f"step {step}: eval summary {key} expected {expected!r}, "
                f"got {summary.get(key)!r}"
            )
    exit_code_path = eval_dir / "exit_code.txt"
    if not exit_code_path.is_file() or exit_code_path.read_text().strip() != "0":
        issues.append(f"step {step}: eval exit_code.txt is missing or nonzero")
    if set(tasks) != EXPECTED_TASKS:
        issues.append(f"step {step}: eval task set is {sorted(tasks)}")
    for task, values in tasks.items():
        examples = int(values.get("num_examples", 0))
        rollouts = int(values.get("total_rollouts", 0))
        expected_examples = EXPECTED_TASK_EXAMPLES.get(task)
        if examples != expected_examples:
            issues.append(
                f"step {step} {task}: expected {expected_examples} examples, got {examples}"
            )
        if rollouts != examples * 8:
            issues.append(
                f"step {step} {task}: {rollouts} rollouts for {examples} examples"
            )
    total_rollouts = sum(
        int(values.get("total_rollouts", 0)) for values in tasks.values()
    )
    if total_rollouts != sum(EXPECTED_TASK_EXAMPLES.values()) * 8:
        issues.append(f"step {step}: expected 5144 total rollouts, got {total_rollouts}")

    config_path = eval_dir / "outputs" / "eval_config.json"
    if not config_path.is_file():
        issues.append(f"step {step}: missing eval_config.json")
        config = {}
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = {
        "n": 8,
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 16384,
        "eval_seed": 21,
        "rollout_seeds": list(range(21, 29)),
        "grader": "verl",
        "enable_thinking": False,
        "model_path": str(
            run_dir
            / "checkpoints"
            / f"global_step_{step}"
            / "actor"
            / "huggingface"
        ),
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            issues.append(
                f"step {step}: eval config {key} expected {expected!r}, "
                f"got {config.get(key)!r}"
            )

    expected_seeds = set(range(21, 29))
    for task, expected_examples in EXPECTED_TASK_EXAMPLES.items():
        graded_path = eval_dir / "outputs" / f"{task}_graded.jsonl"
        if not graded_path.is_file():
            issues.append(f"step {step} {task}: missing graded JSONL")
            continue
        seeds_by_example: dict[str, list[int]] = defaultdict(list)
        for line in graded_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seeds_by_example[str(row["example_id"])].append(int(row["seed"]))
        if len(seeds_by_example) != expected_examples:
            issues.append(
                f"step {step} {task}: graded JSONL has {len(seeds_by_example)} examples"
            )
        bad_examples = [
            example_id
            for example_id, seeds in seeds_by_example.items()
            if len(seeds) != 8 or set(seeds) != expected_seeds
        ]
        if bad_examples:
            issues.append(
                f"step {step} {task}: {len(bad_examples)} examples lack seeds 21-28"
            )
    return issues


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    checkpoint_steps = parse_step_list(args.checkpoint_steps)
    eval_steps = [] if args.skip_eval else parse_step_list(args.eval_steps)
    expected_diagnostic_steps = (
        parse_step_list(args.expected_diagnostic_steps)
        if args.expected_diagnostic_steps
        else EXPECTED_DIAGNOSTIC_STEPS
    )
    issues: list[str] = []
    warnings: list[str] = []

    card_path = run_dir / "run_card.json"
    if not card_path.is_file():
        issues.append("missing run_card.json")
        card = {}
    else:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    expected_card = expected_run_card(
        args.variant,
        total_training_steps=args.expected_total_training_steps,
        diag_interval=args.expected_diag_interval,
        resume_mode=args.expected_resume_mode,
    )
    for key, expected in expected_card.items():
        if card.get(key) != expected:
            issues.append(f"run_card {key}: expected {expected!r}, got {card.get(key)!r}")
    if card.get("source_commit") in (None, "unknown"):
        issues.append("run_card source_commit is missing")
    if (
        args.expected_source_commit is not None
        and card.get("source_commit") != args.expected_source_commit
    ):
        issues.append(
            f"run_card source_commit: expected {args.expected_source_commit!r}, "
            f"got {card.get('source_commit')!r}"
        )
    if (
        args.expected_resume_from_path is not None
        and card.get("resume_from_path") != args.expected_resume_from_path
    ):
        issues.append(
            "run_card resume_from_path: expected "
            f"{args.expected_resume_from_path!r}, got {card.get('resume_from_path')!r}"
        )
    for key, expected_suffix in (
        ("student_model", args.expected_student_model_suffix),
        ("teacher_model", args.expected_teacher_model_suffix),
    ):
        if expected_suffix and not str(card.get(key, "")).endswith(expected_suffix):
            issues.append(
                f"run_card {key}: expected suffix {expected_suffix!r}, got {card.get(key)!r}"
            )

    for required in (
        "artifact_hashes.sha256",
        "script_hashes.sha256",
        "data_manifest.json",
        "revisiting_opd_manifest_check.txt",
        "env.txt",
    ):
        if not (run_dir / required).is_file():
            issues.append(f"missing {required}")
    issues.extend(
        artifact_hash_issues(
            run_dir / "artifact_hashes.sha256",
            args.expected_train_sha256,
        )
    )
    exit_code_path = run_dir / "exit_code.txt"
    if not exit_code_path.is_file() or exit_code_path.read_text().strip() != "0":
        issues.append("training exit_code.txt is missing or nonzero")
    manifest_check = run_dir / "revisiting_opd_manifest_check.txt"
    if manifest_check.is_file():
        manifest_lines = manifest_check.read_text(encoding="utf-8").splitlines()
        if len(manifest_lines) != 1248 or any(not line.endswith(": OK") for line in manifest_lines):
            issues.append("revisiting_opd runtime manifest is incomplete or contains failures")
    env_path = run_dir / "env.txt"
    if env_path.is_file():
        env_text = env_path.read_text(encoding="utf-8")
        for version_key in ("torch=", "transformers=", "vllm="):
            if version_key not in env_text:
                issues.append(f"env.txt missing {version_key} version")

    diagnostic_paths = sorted((run_dir / "diagnostics").glob("step_*.npz"))
    diagnostic_steps = [int(path.stem.split("_")[-1]) for path in diagnostic_paths]
    if diagnostic_steps != expected_diagnostic_steps:
        missing = sorted(set(expected_diagnostic_steps) - set(diagnostic_steps))
        extra = sorted(set(diagnostic_steps) - set(expected_diagnostic_steps))
        issues.append(f"diagnostic steps mismatch; missing={missing}, extra={extra}")

    scalar_path = run_dir / "diagnostics" / "scalars.jsonl"
    scalar_steps = []
    if scalar_path.is_file():
        latest = {}
        for line in scalar_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[int(record["step"])] = record
        scalar_steps = sorted(latest)
        issues.extend(numerical_scalar_issues(list(latest.values())))
    else:
        issues.append("missing diagnostics/scalars.jsonl")
    if scalar_steps != expected_diagnostic_steps:
        issues.append("scalar JSONL steps do not match expected diagnostic schedule")

    for step in checkpoint_steps:
        issues.extend(checkpoint_issues(run_dir, step, world_size=args.world_size))
    for step in eval_steps:
        issues.extend(eval_issues(run_dir, step))

    result = {
        "passed": not issues,
        "run_dir": str(run_dir),
        "source_commit": card.get("source_commit"),
        "diagnostic_steps": diagnostic_steps,
        "checkpoint_steps": checkpoint_steps,
        "eval_steps": eval_steps,
        "issues": issues,
        "warnings": warnings,
    }
    output_path = run_dir / "acceptance.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
