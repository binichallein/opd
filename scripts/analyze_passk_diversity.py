#!/usr/bin/env python3
"""Analyze pass@k predictions with robust grading and diversity metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from regrade_predictions import grade_row, normalize_text_answer, pred_number_boxed_or_last, strip_special


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def response_hash(text: str) -> str:
    return hashlib.sha1(strip_special(text).strip().encode("utf-8")).hexdigest()


def final_answer_key(row: dict[str, Any]) -> str:
    response = str(row.get("response", ""))
    num = pred_number_boxed_or_last(response)
    if num is not None:
        return f"num:{num}"
    norm = normalize_text_answer(response)
    return f"text:{norm}" if norm else "empty:"


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(strip_special(text).lower()))


def mean_pairwise_jaccard_distance(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    sets = [token_set(text) for text in texts]
    vals: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                vals.append(0.0)
            else:
                vals.append(1.0 - (len(sets[i] & sets[j]) / len(union)))
    return sum(vals) / len(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-jsonl", required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.preds_jsonl))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["example_index"])].append(row)

    per_example: list[dict[str, Any]] = []
    total_correct = 0
    total_rollouts = 0
    total_lengths = 0
    hit_max = 0
    for ex_i, items in sorted(grouped.items()):
        grades = [grade_row(row, args.dataset) for row in items]
        primary = [bool(g["primary_correct"]) for g in grades]
        responses = [str(row.get("response", "")) for row in items]
        lengths = [int(row.get("completion_tokens", 0)) for row in items]
        answer_keys = {final_answer_key(row) for row in items}
        response_hashes = {response_hash(response) for response in responses}
        total_correct += sum(primary)
        total_rollouts += len(items)
        total_lengths += sum(lengths)
        hit_max += sum(length >= args.max_new_tokens for length in lengths)
        per_example.append(
            {
                "example_index": ex_i,
                "n": len(items),
                "pass_at_k": any(primary),
                "mean_correct": sum(primary) / max(1, len(primary)),
                "unique_answers": len(answer_keys),
                "unique_response_hashes": len(response_hashes),
                "pairwise_token_jaccard_distance": mean_pairwise_jaccard_distance(responses),
                "avg_completion_tokens": sum(lengths) / max(1, len(lengths)),
                "hit_max_rate": sum(length >= args.max_new_tokens for length in lengths) / max(1, len(lengths)),
            }
        )

    n_examples = len(per_example)
    summary = {
        "model": args.model_name,
        "dataset": args.dataset,
        "preds_jsonl": args.preds_jsonl,
        "examples": n_examples,
        "rollouts": total_rollouts,
        "n": total_rollouts / max(1, n_examples),
        "robust_mean_score": total_correct / max(1, total_rollouts),
        "robust_pass_at_k": sum(item["pass_at_k"] for item in per_example) / max(1, n_examples),
        "avg_unique_answers": sum(item["unique_answers"] for item in per_example) / max(1, n_examples),
        "avg_unique_answer_rate": sum(item["unique_answers"] / max(1, item["n"]) for item in per_example) / max(1, n_examples),
        "avg_unique_response_rate": sum(item["unique_response_hashes"] / max(1, item["n"]) for item in per_example) / max(1, n_examples),
        "avg_pairwise_token_jaccard_distance": sum(item["pairwise_token_jaccard_distance"] for item in per_example)
        / max(1, n_examples),
        "avg_completion_tokens": total_lengths / max(1, total_rollouts),
        "hit_max_token_rate": hit_max / max(1, total_rollouts),
        "grader": "robust primary from regrade_predictions.py; diversity computed from generated samples only",
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
