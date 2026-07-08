#!/usr/bin/env python3
"""Generate and grade pass@k for GSM8K/MATH-style boxed answers."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def render_prompt(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize_answer(text: str) -> str:
    text = text.strip()
    if "####" in text:
        text = text.split("####")[-1]
    boxed = BOX_RE.findall(text)
    if boxed:
        text = boxed[-1]
    text = text.strip().strip(".")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace(" ", "")
    text = text.replace("$", "")
    return text.lower()


def is_correct(response: str, answer: str) -> bool:
    pred = normalize_answer(response)
    gold = normalize_answer(answer)
    if pred == gold:
        return True
    if not pred or not gold:
        return False
    return pred in {gold, "{" + gold + "}"} or gold in {pred, "{" + pred + "}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = load_jsonl(Path(args.data_jsonl), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    ).eval()

    grouped: dict[int, list[bool]] = defaultdict(list)
    lengths: list[int] = []
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_jsonl).open("w", encoding="utf-8") as f:
        for ex_i, row in enumerate(rows):
            prompt = render_prompt(tokenizer, row["prompt"], args.enable_thinking)
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=args.max_prompt_tokens).to(device)
            prompt_len = int(enc["input_ids"].shape[1])
            for sample_i in range(args.n):
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                completion = out[0, prompt_len:]
                response = tokenizer.decode(completion, skip_special_tokens=False)
                correct = is_correct(response, str(row.get("answer", "")))
                grouped[ex_i].append(correct)
                lengths.append(int(completion.numel()))
                f.write(
                    json.dumps(
                        {
                            "example_index": ex_i,
                            "sample_index": sample_i,
                            "id": row.get("id"),
                            "source": row.get("source"),
                            "answer": row.get("answer"),
                            "correct": correct,
                            "completion_tokens": int(completion.numel()),
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    total = sum(len(v) for v in grouped.values())
    correct = sum(sum(v) for v in grouped.values())
    passk = sum(any(v) for v in grouped.values()) / max(1, len(grouped))
    summary = {
        "model_dir": args.model_dir,
        "data_jsonl": args.data_jsonl,
        "examples": len(grouped),
        "n": args.n,
        "rollouts": total,
        "mean_score": correct / max(1, total),
        "pass_at_k": passk,
        "avg_completion_tokens": sum(lengths) / max(1, len(lengths)),
        "args": vars(args),
        "grader": "simple boxed/exact normalizer; use for pilot comparisons, not final theorem-grade reporting",
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
