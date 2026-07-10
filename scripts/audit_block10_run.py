#!/usr/bin/env python3
"""Audit one completed Block10 diagnostic run before cross-host analysis."""

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
    return parser.parse_args()


def checkpoint_issues(run_dir: Path, step: int) -> list[str]:
    root = run_dir / "checkpoints" / f"global_step_{step}"
    actor = root / "actor"
    issues = []
    if not (root / "data.pt").is_file():
        issues.append(f"step {step}: missing data.pt")
    for prefix in ("model", "optim", "extra_state"):
        files = list(actor.glob(f"{prefix}_world_size_4_rank_*.pt"))
        if len(files) != 4 or any(path.stat().st_size == 0 for path in files):
            issues.append(f"step {step}: expected four nonempty {prefix} shards")
    return issues


def eval_issues(run_dir: Path, step: int) -> list[str]:
    summary_path = run_dir / f"eval_step_{step}_n8" / "outputs" / "summary.json"
    if not summary_path.is_file():
        return [f"step {step}: missing n=8 eval summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = summary.get("tasks", {})
    issues = []
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
    issues: list[str] = []
    warnings: list[str] = []

    card_path = run_dir / "run_card.json"
    if not card_path.is_file():
        issues.append("missing run_card.json")
        card = {}
    else:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    expected_card = {
        "variant": "block10_mean",
        "seed": 21,
        "data_seed": 21,
        "rollout_seed": 21,
        "environment_seed": 21,
        "total_training_steps": 200,
        "opd_diagnostics": True,
        "opd_diag_interval": 5,
        "opd_diag_topk": 16,
        "rollout_gpu_memory_utilization": 0.6,
    }
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

    for step in EXPECTED_CHECKPOINT_STEPS:
        issues.extend(checkpoint_issues(run_dir, step))
    for step in EXPECTED_EVAL_STEPS:
        issues.extend(eval_issues(run_dir, step))

    result = {
        "passed": not issues,
        "run_dir": str(run_dir),
        "source_commit": card.get("source_commit"),
        "diagnostic_steps": diagnostic_steps,
        "checkpoint_steps": EXPECTED_CHECKPOINT_STEPS,
        "eval_steps": EXPECTED_EVAL_STEPS,
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
