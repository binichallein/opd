#!/usr/bin/env python3
"""Summarize block advantage normalization experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNS = ["block3_sum", "block3_mean", "block3_mixed_lam05"]
TRAIN_KEYS = [
    "loss",
    "grad_norm",
    "sec",
    "completion_tokens",
    "supervised_tokens",
    "supervision_units",
    "supervised_token_rate",
    "supervision_unit_rate",
    "advantage_mean",
    "positive_advantage_rate",
    "raw_advantage_mean",
    "overlap_mean",
    "sampled_in_teacher_rate",
    "teacher_margin_mean",
    "teacher_entropy_mean",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_train(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"present": False}
    summary: dict[str, Any] = {
        "present": True,
        "last_step": rows[-1].get("step"),
        "steps": len(rows),
        "block_advantage_mode": rows[-1].get("block_advantage_mode"),
    }
    for key in TRAIN_KEYS:
        values = [float(row[key]) for row in rows if key in row and row[key] is not None]
        if values:
            summary[f"{key}_last"] = values[-1]
            summary[f"{key}_mean"] = sum(values) / len(values)
            summary[f"{key}_max"] = max(values)
    return summary


def summarize_eval(run_dir: Path, dataset: str) -> dict[str, Any]:
    regrade = load_json(run_dir / f"{dataset}_regrade_summary.json")
    if regrade:
        return {
            "present": True,
            "source": "regrade",
            "examples": regrade.get("examples"),
            "score": regrade.get("metrics", {}).get("primary_correct"),
            "metrics": regrade.get("metrics", {}),
        }
    online = load_json(run_dir / f"{dataset}_summary.json")
    if online:
        return {
            "present": True,
            "source": "online",
            "examples": online.get("examples"),
            "score": online.get("mean_score"),
            "pass_at_k": online.get("pass_at_k"),
        }
    return {"present": False}


def summarize_run(run_root: Path, name: str) -> dict[str, Any]:
    run_dir = run_root / name
    return {
        "run_dir": str(run_dir),
        "train": summarize_train(load_jsonl(run_dir / "train_metrics.jsonl")),
        "gsm8k": summarize_eval(run_dir, "gsm8k"),
        "math500": summarize_eval(run_dir, "math500"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summary = {
        "run_root": str(run_root),
        "runs": {name: summarize_run(run_root, name) for name in RUNS},
        "decision_rule": (
            "mean or mixed is useful if it matches/exceeds block3_sum quality "
            "while reducing grad_norm mean/max."
        ),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
