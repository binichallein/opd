#!/usr/bin/env python3
"""Prepare public math datasets for clean-room OPD experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import get_dataset_config_names, load_dataset


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gsm8k_rows(split: str, limit: int | None) -> list[dict[str, Any]]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for i, item in enumerate(ds):
        rows.append(
            {
                "id": f"gsm8k/{split}/{i}",
                "source": "openai/gsm8k",
                "split": split,
                "problem": item["question"],
                "prompt": item["question"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
                "answer": item["answer"],
            }
        )
    return rows


def math500_rows(limit: int | None) -> list[dict[str, Any]]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for i, item in enumerate(ds):
        answer = item.get("answer", item.get("solution", ""))
        rows.append(
            {
                "id": f"math500/test/{i}",
                "source": "HuggingFaceH4/MATH-500",
                "split": "test",
                "problem": item["problem"],
                "prompt": item["problem"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
                "answer": answer,
            }
        )
    return rows


def hendrycks_math_train_rows(limit_per_config: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = get_dataset_config_names("EleutherAI/hendrycks_math")
    for config in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="train")
        if limit_per_config is not None:
            ds = ds.select(range(min(limit_per_config, len(ds))))
        for i, item in enumerate(ds):
            rows.append(
                {
                    "id": f"hendrycks_math/{config}/train/{i}",
                    "source": f"EleutherAI/hendrycks_math/{config}",
                    "split": "train",
                    "problem": item["problem"],
                    "prompt": item["problem"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
                    "answer": item.get("solution", ""),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gsm8k-train-limit", type=int, default=2000)
    parser.add_argument("--math-train-limit-per-config", type=int, default=300)
    parser.add_argument("--eval-limit", type=int, default=0, help="0 means full eval splits")
    args = parser.parse_args()

    out = Path(args.output_dir)
    eval_limit = args.eval_limit or None

    train_rows = []
    train_rows.extend(gsm8k_rows("train", args.gsm8k_train_limit))
    try:
        train_rows.extend(hendrycks_math_train_rows(args.math_train_limit_per_config))
        math_train_status = "ok"
    except Exception as exc:
        math_train_status = f"failed: {type(exc).__name__}: {exc}"

    eval_gsm8k = gsm8k_rows("test", eval_limit)
    eval_math500 = math500_rows(eval_limit)

    write_jsonl(out / "train_prompts.jsonl", train_rows)
    write_jsonl(out / "eval_gsm8k.jsonl", eval_gsm8k)
    write_jsonl(out / "eval_math500.jsonl", eval_math500)

    manifest = {
        "train_prompts": str(out / "train_prompts.jsonl"),
        "eval_gsm8k": str(out / "eval_gsm8k.jsonl"),
        "eval_math500": str(out / "eval_math500.jsonl"),
        "counts": {
            "train_prompts": len(train_rows),
            "eval_gsm8k": len(eval_gsm8k),
            "eval_math500": len(eval_math500),
        },
        "sources": [
            "openai/gsm8k",
            "HuggingFaceH4/MATH-500",
            "EleutherAI/hendrycks_math",
        ],
        "math_train_status": math_train_status,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
