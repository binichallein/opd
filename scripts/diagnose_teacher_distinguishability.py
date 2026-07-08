#!/usr/bin/env python3
"""Teacher local distinguishability diagnostics on student rollouts.

The script is clean-room and only depends on explicitly provided model dirs.
It generates rollouts with the student, then scores the same prefixes with
multiple official teacher candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
    if not rows:
        raise ValueError(f"empty jsonl: {path}")
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


def parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"teacher must be NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"teacher must be NAME=PATH, got: {value}")
    return name, path


def topk_entropy(top_logp: torch.Tensor) -> torch.Tensor:
    probs = top_logp.exp()
    mass = probs.sum(dim=-1).clamp_min(1e-8)
    norm = probs / mass.unsqueeze(-1)
    return -(norm * norm.clamp_min(1e-8).log()).sum(dim=-1)


@torch.no_grad()
def score_sequence(model, input_ids: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    sampled = logp.gather(-1, labels.unsqueeze(-1)).squeeze(0).squeeze(-1)
    top_logp, top_ids = torch.topk(logp.squeeze(0), k=top_k, dim=-1)
    return {
        "sampled_logp": sampled,
        "top_ids": top_ids,
        "top_logp": top_logp,
        "top1_ids": top_ids[:, 0],
        "margin": top_logp[:, 0] - top_logp[:, 1],
        "entropy": topk_entropy(top_logp),
    }


def slice_completion(score: dict[str, torch.Tensor], prompt_len: int, completion_len: int) -> dict[str, torch.Tensor]:
    start = max(0, prompt_len - 1)
    stop = start + completion_len
    return {key: value[start:stop] for key, value in score.items()}


def topk_overlap(a_ids: torch.Tensor, b_ids: torch.Tensor) -> torch.Tensor:
    return a_ids.unsqueeze(2).eq(b_ids.unsqueeze(1)).any(dim=2).float().mean(dim=1)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 1e-8:
        return float("nan")
    return float((x * y).sum().div(denom).item())


def safe_mean(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return sum(vals) / max(1, len(vals))


def tensor_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(x.float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-dir", required=True)
    parser.add_argument("--teacher", action="append", required=True, help="NAME=MODEL_DIR. May be repeated.")
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--limit", type=int, default=96)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    teacher_specs = [parse_named_path(v) for v in args.teacher]
    teacher_names = [name for name, _ in teacher_specs]
    if len(set(teacher_names)) != len(teacher_names):
        raise ValueError(f"duplicate teacher names: {teacher_names}")

    tokenizer = AutoTokenizer.from_pretrained(args.student_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        args.student_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    ).eval()
    teachers: dict[str, Any] = {}
    for name, model_dir in teacher_specs:
        if Path(model_dir).resolve() == Path(args.student_dir).resolve():
            teachers[name] = student
        else:
            teachers[name] = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map={"": device},
            ).eval()

    rows = load_jsonl(Path(args.data_jsonl), args.limit)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, list[float]] = defaultdict(list)
    total_completion_tokens = 0

    with Path(args.output_jsonl).open("w", encoding="utf-8") as out_f:
        for ex_i, row in enumerate(rows):
            prompt_text = render_prompt(tokenizer, row["prompt"], args.enable_thinking)
            enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_prompt_tokens,
            ).to(device)
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
            completion_len = int(gen.shape[1] - prompt_len)
            if completion_len <= 0:
                continue
            gen = gen[:, : prompt_len + completion_len]
            sampled_ids = gen[:, 1:].squeeze(0)[prompt_len - 1 : prompt_len - 1 + completion_len]
            total_completion_tokens += completion_len

            student_score = slice_completion(score_sequence(student, gen, args.top_k), prompt_len, completion_len)
            teacher_scores = {
                name: slice_completion(score_sequence(model, gen, args.top_k), prompt_len, completion_len)
                for name, model in teachers.items()
            }

            rec: dict[str, Any] = {
                "example_index": ex_i,
                "id": row.get("id"),
                "source": row.get("source"),
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_len,
                "completion_preview": tokenizer.decode(
                    gen[0, prompt_len : prompt_len + min(128, completion_len)],
                    skip_special_tokens=False,
                ),
                "teacher_vs_student": {},
                "teacher_pairs": {},
            }

            s_logp = student_score["sampled_logp"]
            s_top_ids = student_score["top_ids"]
            for name, score in teacher_scores.items():
                t_logp = score["sampled_logp"]
                advantage = (t_logp - s_logp).clamp(-20, 20)
                overlap = topk_overlap(s_top_ids, score["top_ids"])
                sampled_in_teacher = score["top_ids"].eq(sampled_ids.unsqueeze(-1)).any(dim=1).float()
                metrics = {
                    "sampled_logp_mean": tensor_mean(t_logp),
                    "advantage_mean": tensor_mean(advantage),
                    "advantage_abs_mean": tensor_mean(advantage.abs()),
                    "positive_advantage_rate": tensor_mean((advantage > 0).float()),
                    "topk_overlap_with_student": tensor_mean(overlap),
                    "sampled_in_teacher_topk_rate": tensor_mean(sampled_in_teacher),
                    "top1_agreement_with_student": tensor_mean((score["top1_ids"] == student_score["top1_ids"]).float()),
                    "margin_mean": tensor_mean(score["margin"]),
                    "entropy_mean": tensor_mean(score["entropy"]),
                }
                rec["teacher_vs_student"][name] = metrics
                for key, value in metrics.items():
                    stats[f"teacher.{name}.{key}"].append(float(value))

            for i, a in enumerate(teacher_names):
                for b in teacher_names[i + 1 :]:
                    a_score = teacher_scores[a]
                    b_score = teacher_scores[b]
                    a_adv = (a_score["sampled_logp"] - s_logp).clamp(-20, 20)
                    b_adv = (b_score["sampled_logp"] - s_logp).clamp(-20, 20)
                    pair_key = f"{a}__vs__{b}"
                    metrics = {
                        "sampled_logp_abs_diff": tensor_mean((a_score["sampled_logp"] - b_score["sampled_logp"]).abs()),
                        "sampled_logp_corr": pearson(a_score["sampled_logp"], b_score["sampled_logp"]),
                        "topk_overlap": tensor_mean(topk_overlap(a_score["top_ids"], b_score["top_ids"])),
                        "top1_agreement": tensor_mean((a_score["top1_ids"] == b_score["top1_ids"]).float()),
                        "advantage_sign_disagreement": tensor_mean(((a_adv > 0) != (b_adv > 0)).float()),
                        "advantage_abs_diff": tensor_mean((a_adv - b_adv).abs()),
                    }
                    rec["teacher_pairs"][pair_key] = metrics
                    for key, value in metrics.items():
                        stats[f"pair.{pair_key}.{key}"].append(float(value))

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "examples": len(rows),
        "scored_completion_tokens": total_completion_tokens,
        "teachers": [{"name": name, "dir": path} for name, path in teacher_specs],
        "means": {key: safe_mean(values) for key, values in sorted(stats.items())},
        "args": vars(args),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
