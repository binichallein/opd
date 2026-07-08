#!/usr/bin/env python3
"""Offline regrading for saved clean-room GSM8K/MATH predictions.

The online evaluator intentionally used a simple boxed/exact normalizer. This
script keeps that original signal and adds more explicit grading views without
rerunning generation.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?")
SPECIAL_RE = re.compile(r"<\\|[^>]+\\|>|<\\/?s>|<\\|endoftext\\|>")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def strip_special(text: str) -> str:
    return SPECIAL_RE.sub(" ", text)


def boxed_values(text: str) -> list[str]:
    return [x.strip() for x in BOX_RE.findall(text)]


def normalize_text_answer(text: str) -> str:
    text = strip_special(text)
    if "####" in text:
        text = text.split("####")[-1]
    boxes = boxed_values(text)
    if boxes:
        text = boxes[-1]
    text = text.strip().strip(".")
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "\\ ": "",
        "$": "",
        "\\text": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", "", text)
    text = text.replace("{", "").replace("}", "")
    return text.lower()


def decimal_from_number_token(token: str) -> Decimal | None:
    token = token.strip().replace(",", "")
    if "/" in token and token.count("/") == 1:
        left, right = token.split("/")
        try:
            denom = Decimal(right)
            if denom == 0:
                return None
            return Decimal(left) / denom
        except InvalidOperation:
            return None
    try:
        return Decimal(token)
    except InvalidOperation:
        return None


def extract_numbers(text: str) -> list[Decimal]:
    vals: list[Decimal] = []
    for match in NUM_RE.findall(strip_special(text)):
        val = decimal_from_number_token(match)
        if val is not None:
            vals.append(val)
    return vals


def gold_number(answer: str) -> Decimal | None:
    text = answer.split("####")[-1] if "####" in answer else answer
    nums = extract_numbers(text)
    return nums[-1] if nums else None


def pred_number_boxed_or_last(response: str) -> Decimal | None:
    boxes = boxed_values(response)
    if boxes:
        nums = extract_numbers(boxes[-1])
        if nums:
            return nums[-1]
    text = strip_special(response)
    lower = text.lower()
    for marker in ["final answer", "answer:", "answer is", "therefore"]:
        pos = lower.rfind(marker)
        if pos >= 0:
            nums = extract_numbers(text[pos:])
            if nums:
                return nums[-1]
    nums = extract_numbers(text)
    return nums[-1] if nums else None


def numeric_equal(a: Decimal | None, b: Decimal | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= Decimal("1e-9")


def exact_match(response: str, answer: str) -> bool:
    pred = normalize_text_answer(response)
    gold = normalize_text_answer(answer)
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    return pred in {gold, "{" + gold + "}"} or gold in {pred, "{" + pred + "}"}


def boxed_exact_match(response: str, answer: str) -> bool:
    boxes = boxed_values(response)
    if not boxes:
        return False
    return normalize_text_answer(boxes[-1]) == normalize_text_answer(answer)


def grade_row(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    response = str(row.get("response", ""))
    answer = str(row.get("answer", ""))
    original = bool(row.get("correct", False))
    strict_exact = exact_match(response, answer)
    strict_boxed = boxed_exact_match(response, answer)
    g_num = gold_number(answer)
    p_num = pred_number_boxed_or_last(response)
    numeric = numeric_equal(g_num, p_num)
    if dataset == "gsm8k":
        primary = numeric
    else:
        primary = strict_exact or strict_boxed
        if g_num is not None and p_num is not None and normalize_text_answer(answer).replace(".", "", 1).isdigit():
            primary = primary or numeric
    return {
        "original_correct": original,
        "strict_exact": strict_exact,
        "boxed_exact": strict_boxed,
        "numeric_boxed_or_last": numeric,
        "primary_correct": primary,
        "gold_number": str(g_num) if g_num is not None else None,
        "pred_number": str(p_num) if p_num is not None else None,
    }


def summarize(rows: list[dict[str, Any]], dataset: str, model_name: str, preds_jsonl: Path) -> dict[str, Any]:
    graded = [grade_row(row, dataset) for row in rows]
    n = len(graded)
    keys = ["original_correct", "strict_exact", "boxed_exact", "numeric_boxed_or_last", "primary_correct"]
    summary = {
        "model": model_name,
        "dataset": dataset,
        "preds_jsonl": str(preds_jsonl),
        "examples": n,
        "metrics": {key: sum(bool(g[key]) for g in graded) / max(1, n) for key in keys},
        "disagreements": {
            "original_vs_primary": sum(bool(g["original_correct"]) != bool(g["primary_correct"]) for g in graded),
            "strict_exact_vs_primary": sum(bool(g["strict_exact"]) != bool(g["primary_correct"]) for g in graded),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-jsonl", required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--graded-jsonl", default="")
    args = parser.parse_args()

    preds_path = Path(args.preds_jsonl)
    rows = load_jsonl(preds_path)
    graded = [grade_row(row, args.dataset) for row in rows]
    summary = summarize(rows, args.dataset, args.model_name, preds_path)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.graded_jsonl:
        out = Path(args.graded_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row, grade in zip(rows, graded):
                item = {
                    "example_index": row.get("example_index"),
                    "id": row.get("id"),
                    **grade,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
