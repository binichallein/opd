#!/usr/bin/env python3
"""Audit one completed OPD diagnostic run before analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_DIAGNOSTIC_STEPS = [1, *range(5, 201, 5)]
EXPECTED_CHECKPOINT_STEPS = [40, 50, 60, 80, 100, 200]
EXPECTED_EVAL_STEPS = [50, 100, 200]
EXPECTED_TASKS = {"math500", "aime24", "aime25", "amc23"}


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
    return parser.parse_args()


def parse_step_list(value: str) -> list[int]:
    """Parse a comma-separated milestone list into sorted unique positive steps."""
    if not value.strip():
        return []
    steps = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if any(step <= 0 for step in steps):
        raise ValueError("milestone steps must be positive")
    return steps


def expected_run_card(variant: str) -> dict[str, object]:
    """Return the invariant run-card fields required for diagnostic experiments."""
    return {
        "variant": variant,
        "seed": 21,
        "data_seed": 21,
        "rollout_seed": 21,
        "environment_seed": 21,
        "total_training_steps": 200,
        "test_freq": -1,
        "opd_diagnostics": True,
        "opd_diag_interval": 5,
        "opd_diag_topk": 16,
        "opd_diag_position_stride": 1,
        "rollout_gpu_memory_utilization": 0.6,
        "ref_log_prob_micro_batch_size_per_gpu": 1,
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
        if len(files) != world_size or any(path.stat().st_size == 0 for path in files):
            issues.append(
                f"step {step}: expected {world_size} nonempty {prefix} shards"
            )
    return issues


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
        if examples <= 0 or rollouts != examples * 8:
            issues.append(
                f"step {step} {task}: {rollouts} rollouts for {examples} examples"
            )
    return issues


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    checkpoint_steps = parse_step_list(args.checkpoint_steps)
    eval_steps = [] if args.skip_eval else parse_step_list(args.eval_steps)
    issues: list[str] = []
    warnings: list[str] = []

    card_path = run_dir / "run_card.json"
    if not card_path.is_file():
        issues.append("missing run_card.json")
        card = {}
    else:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    expected_card = expected_run_card(args.variant)
    for key, expected in expected_card.items():
        if card.get(key) != expected:
            issues.append(f"run_card {key}: expected {expected!r}, got {card.get(key)!r}")
    if card.get("source_commit") in (None, "unknown"):
        issues.append("run_card source_commit is missing")

    for required in (
        "artifact_hashes.sha256",
        "script_hashes.sha256",
        "data_manifest.json",
        "revisiting_opd_manifest_check.txt",
        "env.txt",
    ):
        if not (run_dir / required).is_file():
            issues.append(f"missing {required}")
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
    if diagnostic_steps != EXPECTED_DIAGNOSTIC_STEPS:
        missing = sorted(set(EXPECTED_DIAGNOSTIC_STEPS) - set(diagnostic_steps))
        extra = sorted(set(diagnostic_steps) - set(EXPECTED_DIAGNOSTIC_STEPS))
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
        for step, record in latest.items():
            for key, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isfinite(float(value)):
                        warnings.append(f"step {step}: nonfinite scalar {key}={value}")
    else:
        issues.append("missing diagnostics/scalars.jsonl")
    if scalar_steps != EXPECTED_DIAGNOSTIC_STEPS:
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
