#!/usr/bin/env python3
"""Prepare DAPO-Math-17K data for the revisiting_opd math environment.

The output format matches external/revisiting_opd/examples/data_preprocess/
math_opd_process.py, but without hard-coded paths. The validation parquet is
the union of AIME24, MATH500, and optional extra paper eval sets so one final
validation pass reports data sources separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets, load_dataset


DAPO_URL = "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
AIME24_URL = "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"


def parquet_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def prompt_content(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
    if isinstance(prompt, dict) and "content" in prompt:
        return str(prompt["content"])
    raise TypeError(f"Unsupported prompt format: {type(prompt)!r}")


def clean_prompt(prompt: str) -> str:
    cleaned_lines: list[str] = []
    for line in prompt.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Solve the following math problem"):
            continue
        if stripped.startswith("The last line of your response should be of the form"):
            continue
        if stripped.startswith("Remember to put your answer on its own line"):
            continue
        if stripped.startswith("$Answer (without quotes) where $Answer"):
            continue
        cleaned_lines.append(line.rstrip())
    question = "\n".join(cleaned_lines).strip()
    if not question:
        raise ValueError("Prompt became empty after cleaning")
    return question


def ground_truth_from_reward_model(reward_model: Any) -> str:
    if isinstance(reward_model, dict):
        if "ground_truth" in reward_model:
            return str(reward_model["ground_truth"])
        if "answer" in reward_model:
            return str(reward_model["answer"])
    raise KeyError(f"Unsupported reward_model format: {reward_model!r}")


def env_row(question: str, answer: str, split: str, source: str, idx: int) -> dict[str, Any]:
    return {
        "data_source": source,
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": str(answer),
        },
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "extra_info": {
            "split": split,
            "index": idx,
            "question": question,
            "answer": str(answer),
        },
        "env_kwargs": {
            "question": question,
            "ground_truth": str(answer),
            "data_source": source,
        },
    }


def eval_json_row(question: str, answer: str, source: str, split: str, idx: int) -> dict[str, Any]:
    prompt = question.strip()
    if "put your final answer" not in prompt:
        prompt = prompt + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    return {
        "id": f"{source}/{split}/{idx}",
        "source": source,
        "split": split,
        "problem": question,
        "prompt": prompt,
        "answer": str(answer),
    }


def dapo_dataset(path: Path, keep_duplicates: bool = False) -> tuple[Dataset, dict[str, Any]]:
    raw = load_dataset("parquet", data_files=str(path), split="train")
    env_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    duplicate_rows = 0
    for raw_idx, example in enumerate(raw):
        question = clean_prompt(prompt_content(example["prompt"]))
        answer = ground_truth_from_reward_model(example["reward_model"])
        key = (question, answer)
        if not keep_duplicates and key in seen:
            duplicate_rows += 1
            continue
        seen.add(key)
        env_rows.append(env_row(question, answer, "train", "dapo-math-17k", len(env_rows)))
    stats = {
        "raw_rows": len(raw),
        "unique_prompt_answer_pairs": len(seen),
        "duplicates_removed": duplicate_rows,
        "keep_duplicates": keep_duplicates,
    }
    return Dataset.from_list(env_rows), stats


def aime24_datasets(path: Path) -> tuple[Dataset, list[dict[str, Any]]]:
    raw = load_dataset("parquet", data_files=str(path), split="train")
    env_rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for example in raw:
        question = clean_prompt(prompt_content(example["prompt"]))
        answer = ground_truth_from_reward_model(example["reward_model"])
        key = (question, answer)
        if key in seen:
            continue
        seen.add(key)
        idx = len(env_rows)
        env_rows.append(env_row(question, answer, "test", "aime24", idx))
        json_rows.append(eval_json_row(question, answer, "aime24", "test", idx))

    return Dataset.from_list(env_rows), json_rows


def math500_datasets() -> tuple[Dataset, list[dict[str, Any]]]:
    raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
    env_rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []
    for idx, example in enumerate(raw):
        question = str(example["problem"]).strip()
        answer = str(example.get("answer", example.get("solution", ""))).strip()
        env_rows.append(env_row(question, answer, "test", "math500", idx))
        json_rows.append(eval_json_row(question, answer, "math500", "test", idx))
    return Dataset.from_list(env_rows), json_rows


def parquet_eval_datasets(path: Path, source: str) -> tuple[Dataset, list[dict[str, Any]]]:
    raw = load_dataset("parquet", data_files=str(path), split="train")
    env_rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for example in raw:
        question = prompt_content(example["prompt"]).strip()
        answer = ground_truth_from_reward_model(example["reward_model"])
        key = (question, answer)
        if key in seen:
            continue
        seen.add(key)
        idx = len(env_rows)
        env_rows.append(env_row(question, answer, "test", source, idx))
        json_rows.append(eval_json_row(question, answer, source, "test", idx))
    return Dataset.from_list(env_rows), json_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument(
        "--train-source-parquet",
        default=None,
        help=(
            "Optional already-prepared DAPO-Math-17K parquet. It is still normalized and, "
            "by default, deduplicated by (question, answer)."
        ),
    )
    parser.add_argument(
        "--keep-train-duplicates",
        action="store_true",
        help="Keep duplicate DAPO train rows. Default is to deduplicate to the 17K prompt pool.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dapo-url", default=DAPO_URL)
    parser.add_argument("--aime24-url", default=AIME24_URL)
    parser.add_argument(
        "--extra-eval-data-dir",
        default=None,
        help=(
            "Optional directory with paper eval parquet files in subdirectories "
            "AIME25/test.parquet and AMC23/test.parquet."
        ),
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else out / "raw"
    out.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_dapo = raw_dir / "dapo-math-17k.parquet"
    raw_aime24 = raw_dir / "aime-2024.parquet"
    if not args.train_source_parquet:
        download(args.dapo_url, raw_dapo, args.overwrite)
    download(args.aime24_url, raw_aime24, args.overwrite)

    train_source = Path(args.train_source_parquet) if args.train_source_parquet else None
    if train_source:
        train, train_stats = dapo_dataset(train_source, keep_duplicates=args.keep_train_duplicates)
    else:
        train, train_stats = dapo_dataset(raw_dapo, keep_duplicates=args.keep_train_duplicates)
    train_rows = len(train)
    aime24, aime_json = aime24_datasets(raw_aime24)
    math500, math_json = math500_datasets()
    eval_datasets: list[Dataset] = [aime24, math500]
    eval_json_rows: dict[str, list[dict[str, Any]]] = {
        "aime24": aime_json,
        "math500": math_json,
    }
    extra_raw_manifest: dict[str, dict[str, Any]] = {}
    if args.extra_eval_data_dir:
        extra_dir = Path(args.extra_eval_data_dir)
        for dirname, source in [("AIME25", "aime25"), ("AMC23", "amc23")]:
            path = extra_dir / dirname / "test.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing extra eval parquet: {path}")
            ds, rows = parquet_eval_datasets(path, source)
            eval_datasets.append(ds)
            eval_json_rows[source] = rows
            extra_raw_manifest[source] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    val = concatenate_datasets(eval_datasets)

    train_path = out / "train.parquet"
    test_path = out / "test.parquet"
    eval_dir = out / "eval_jsonl"
    eval_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(train_path)
    val.to_parquet(test_path)
    for source, rows in eval_json_rows.items():
        write_jsonl(eval_dir / f"{source}.jsonl", rows)

    manifest = {
        "train_parquet": str(train_path),
        "test_parquet": str(test_path),
        "eval_jsonl": {
            source: str(eval_dir / f"{source}.jsonl")
            for source in eval_json_rows
        },
        "raw": {
            "dapo_math_17k": {
                "path": str(train_source or raw_dapo),
                "url": None if train_source else args.dapo_url,
                "source_mode": "copied_train_source_parquet" if train_source else "downloaded_hf_current_parquet",
                "sha256": sha256_file(train_source or raw_dapo),
                **train_stats,
            },
            "aime24": {
                "path": str(raw_aime24),
                "url": args.aime24_url,
                "sha256": sha256_file(raw_aime24),
            },
            "math500": {
                "dataset": "HuggingFaceH4/MATH-500",
                "split": "test",
            },
            **extra_raw_manifest,
        },
        "counts": {
            "train": train_rows,
            "test_total": len(val),
            "aime24": len(aime24),
            "math500": len(math500),
            **{
                source: len(rows)
                for source, rows in eval_json_rows.items()
                if source not in {"aime24", "math500"}
            },
        },
        "sha256": {
            "train_parquet": sha256_file(train_path),
            "test_parquet": sha256_file(test_path),
            **{
                f"{source}_jsonl": sha256_file(eval_dir / f"{source}.jsonl")
                for source in eval_json_rows
            },
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
