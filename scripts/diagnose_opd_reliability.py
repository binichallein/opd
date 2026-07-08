#!/usr/bin/env python3
"""No-training OPD reliability diagnostics with official models."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def topk_entropy(logp: torch.Tensor) -> torch.Tensor:
    probs = logp.exp()
    mass = probs.sum(dim=-1).clamp_min(1e-8)
    norm = probs / mass.unsqueeze(-1)
    return -(norm * norm.clamp_min(1e-8).log()).sum(dim=-1)


@torch.no_grad()
def score_sequence(model, input_ids: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    sampled = logp.gather(-1, labels.unsqueeze(-1)).squeeze(0).squeeze(-1)
    top_logp, top_ids = torch.topk(logp.squeeze(0), k=top_k, dim=-1)
    return sampled, top_ids, top_logp


def safe_mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-dir", required=True)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.student_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = AutoModelForCausalLM.from_pretrained(
        args.student_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    ).eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    ).eval()

    rows = load_jsonl(Path(args.data_jsonl), args.limit)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, list[float]] = defaultdict(list)
    count_tokens = 0

    with Path(args.output_jsonl).open("w", encoding="utf-8") as out_f:
        for ex_i, row in enumerate(rows):
            prompt_text = render_prompt(tokenizer, row["prompt"], args.enable_thinking)
            enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=args.max_prompt_tokens).to(device)
            prompt_len = int(enc["input_ids"].shape[1])
            gen = student.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full = gen[:, : args.max_prompt_tokens + args.max_new_tokens]
            completion_len = int(full.shape[1] - prompt_len)
            if completion_len <= 0:
                continue
            student_sampled, student_top_ids, student_top_logp = score_sequence(student, full, args.top_k)
            teacher_sampled, teacher_top_ids, teacher_top_logp = score_sequence(teacher, full, args.top_k)
            start = max(0, prompt_len - 1)
            stop = start + completion_len

            s_ids = student_top_ids[start:stop]
            t_ids = teacher_top_ids[start:stop]
            s_lp = student_top_logp[start:stop]
            t_lp = teacher_top_logp[start:stop]
            sampled_ids = full[:, 1:].squeeze(0)[start:stop]
            s_sampled = student_sampled[start:stop]
            t_sampled = teacher_sampled[start:stop]

            match = s_ids.unsqueeze(2).eq(t_ids.unsqueeze(1))
            overlap_ratio = match.any(dim=2).float().mean(dim=1)
            sampled_in_teacher = t_ids.eq(sampled_ids.unsqueeze(-1)).any(dim=1)
            teacher_margin = t_lp[:, 0] - t_lp[:, 1]
            student_entropy = topk_entropy(s_lp)
            teacher_entropy = topk_entropy(t_lp)
            advantage = (t_sampled - s_sampled).clamp(-20, 20)
            rel = (
                (overlap_ratio >= 0.20).float()
                * sampled_in_teacher.float()
                * (teacher_margin >= 0.50).float()
            )
            adaptive_lambda = 1.0 + 0.25 * rel
            adaptive_lambda = torch.where(teacher_entropy > 2.0, torch.ones_like(adaptive_lambda), adaptive_lambda)

            rec = {
                "example_index": ex_i,
                "id": row.get("id"),
                "source": row.get("source"),
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_len,
                "mean_overlap_ratio": float(overlap_ratio.mean().item()),
                "sampled_in_teacher_topk_rate": float(sampled_in_teacher.float().mean().item()),
                "mean_teacher_margin": float(teacher_margin.mean().item()),
                "mean_teacher_entropy_topk": float(teacher_entropy.mean().item()),
                "mean_student_entropy_topk": float(student_entropy.mean().item()),
                "mean_advantage": float(advantage.mean().item()),
                "positive_advantage_rate": float((advantage > 0).float().mean().item()),
                "router_keep_rate": float(rel.mean().item()),
                "adaptive_lambda_mean": float(adaptive_lambda.mean().item()),
                "completion_preview": tokenizer.decode(full[0, prompt_len : prompt_len + min(128, completion_len)], skip_special_tokens=False),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            for key, value in rec.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    stats[key].append(float(value))
            count_tokens += completion_len

    summary = {
        "examples": len(rows),
        "scored_completion_tokens": count_tokens,
        "means": {key: safe_mean(values) for key, values in sorted(stats.items())},
        "args": vars(args),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
