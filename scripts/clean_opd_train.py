#!/usr/bin/env python3
"""Clean-room single-GPU OPD training variants.

This script intentionally does not import any previous project code. It is sized
for mechanism-scale experiments with official public models.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty jsonl: {path}")
    return rows


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


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


def token_logprobs(model, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    sampled = logp.gather(-1, labels.unsqueeze(-1)).squeeze(0).squeeze(-1)
    start = max(0, prompt_len - 1)
    completion_len = input_ids.shape[1] - prompt_len
    return sampled[start : start + completion_len]


@torch.no_grad()
def token_logprobs_topk(
    model,
    input_ids: torch.Tensor,
    prompt_len: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    sampled = logp.gather(-1, labels.unsqueeze(-1)).squeeze(0).squeeze(-1)
    top_logp, top_ids = torch.topk(logp.squeeze(0), k=top_k, dim=-1)
    start = max(0, prompt_len - 1)
    completion_len = input_ids.shape[1] - prompt_len
    return (
        sampled[start : start + completion_len],
        top_ids[start : start + completion_len],
        top_logp[start : start + completion_len],
    )


def topk_entropy(top_logp: torch.Tensor) -> torch.Tensor:
    probs = top_logp.exp()
    mass = probs.sum(dim=-1).clamp_min(1e-8)
    norm = probs / mass.unsqueeze(-1)
    return -(norm * norm.clamp_min(1e-8).log()).sum(dim=-1)


def reliability_signals(
    sampled_ids: torch.Tensor,
    student_top_ids: torch.Tensor,
    student_top_logp: torch.Tensor,
    teacher_top_ids: torch.Tensor,
    teacher_top_logp: torch.Tensor,
    overlap_min: float,
    margin_min: float,
    entropy_max: float,
) -> dict[str, torch.Tensor]:
    match = student_top_ids.unsqueeze(2).eq(teacher_top_ids.unsqueeze(1))
    overlap = match.any(dim=2).float().mean(dim=1)
    sampled_in_teacher = teacher_top_ids.eq(sampled_ids.unsqueeze(-1)).any(dim=1).float()
    teacher_margin = teacher_top_logp[:, 0] - teacher_top_logp[:, 1]
    student_entropy = topk_entropy(student_top_logp)
    teacher_entropy = topk_entropy(teacher_top_logp)
    reliable = (
        (overlap >= overlap_min)
        & (sampled_in_teacher > 0)
        & (teacher_margin >= margin_min)
        & (teacher_entropy <= entropy_max)
    ).float()
    return {
        "overlap": overlap,
        "sampled_in_teacher": sampled_in_teacher,
        "teacher_margin": teacher_margin,
        "student_entropy": student_entropy,
        "teacher_entropy": teacher_entropy,
        "reliable": reliable,
    }


def variant_weights(
    variant: str,
    reliable: torch.Tensor,
    router_floor: float,
    exopd_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == "standard":
        return torch.ones_like(reliable), torch.ones_like(reliable)
    if variant == "topk_router":
        return router_floor + (1.0 - router_floor) * reliable, torch.ones_like(reliable)
    if variant == "adaptive_exopd":
        return torch.ones_like(reliable), torch.full_like(reliable, exopd_lambda)
    if variant == "routed_adaptive":
        return router_floor + (1.0 - router_floor) * reliable, 1.0 + (exopd_lambda - 1.0) * reliable
    if variant == "outcome_reweight":
        return torch.ones_like(reliable), torch.ones_like(reliable)
    if variant == "outcome_topk_router":
        return router_floor + (1.0 - router_floor) * reliable, torch.ones_like(reliable)
    if variant == "anchored_standard":
        return torch.ones_like(reliable), torch.ones_like(reliable)
    if variant == "anchored_topk_router":
        return router_floor + (1.0 - router_floor) * reliable, torch.ones_like(reliable)
    if variant == "ref_penalty_standard":
        return torch.ones_like(reliable), torch.ones_like(reliable)
    if variant == "ref_penalty_topk_router":
        return router_floor + (1.0 - router_floor) * reliable, torch.ones_like(reliable)
    raise ValueError(f"unknown variant: {variant}")


def build_supervision_units(
    *,
    weights: torch.Tensor,
    raw_advantage: torch.Tensor,
    lambdas: torch.Tensor,
    signals: dict[str, torch.Tensor],
    current_logp: torch.Tensor,
    token_supervision_stride: int,
    token_supervision_offset: int,
    block_size: int,
    block_advantage_mode: str = "sum",
    block_mix_lambda: float = 0.5,
    reference_logp: torch.Tensor | None = None,
    reference_gap: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Convert token-level OPD statistics into token or block loss units."""
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if block_advantage_mode not in {"sum", "mean", "mixed"}:
        raise ValueError("block_advantage_mode must be one of: sum, mean, mixed")
    if block_mix_lambda < 0.0 or block_mix_lambda > 1.0:
        raise ValueError("block_mix_lambda must satisfy 0 <= lambda <= 1")
    completion_len = int(current_logp.shape[0])
    if block_size > 1 and (token_supervision_stride != 1 or token_supervision_offset != 0):
        raise ValueError("block OPD requires the default dense token supervision stride/offset")

    if block_size == 1:
        mask = (
            torch.arange(completion_len, device=current_logp.device) % int(token_supervision_stride)
        ).eq(int(token_supervision_offset))
        unit_raw_advantage = raw_advantage[mask]
        unit_lambdas = lambdas[mask]
        result: dict[str, Any] = {
            "weights": weights[mask],
            "raw_advantage": unit_raw_advantage,
            "advantage": (unit_raw_advantage * unit_lambdas).detach(),
            "lambdas": unit_lambdas,
            "signals": {key: value[mask] for key, value in signals.items()},
            "current_logp": current_logp[mask],
            "supervised_tokens": int(mask.sum().detach().cpu()),
            "supervision_units": int(mask.sum().detach().cpu()),
        }
        if reference_logp is not None:
            result["reference_logp"] = reference_logp[mask]
        if reference_gap is not None:
            result["reference_gap"] = reference_gap[mask]
        return result

    usable_tokens = (completion_len // block_size) * block_size
    supervision_units = usable_tokens // block_size
    if supervision_units <= 0:
        empty = current_logp[:0]
        result = {
            "weights": empty,
            "raw_advantage": empty,
            "advantage": empty,
            "lambdas": empty,
            "signals": {key: value[:0] for key, value in signals.items()},
            "current_logp": empty,
            "supervised_tokens": 0,
            "supervision_units": 0,
        }
        if reference_logp is not None:
            result["reference_logp"] = empty
        if reference_gap is not None:
            result["reference_gap"] = empty
        return result

    def blocks(x: torch.Tensor) -> torch.Tensor:
        return x[:usable_tokens].reshape(supervision_units, block_size)

    raw_blocks = blocks(raw_advantage)
    lambda_blocks = blocks(lambdas)
    weighted_raw_blocks = raw_blocks * lambda_blocks
    if block_advantage_mode == "mixed":
        block_mean_advantage = weighted_raw_blocks.mean(dim=1, keepdim=True)
        local_advantage = weighted_raw_blocks
        mixed_advantage = (1.0 - block_mix_lambda) * local_advantage + block_mix_lambda * block_mean_advantage
        result = {
            "weights": blocks(weights).reshape(-1),
            "raw_advantage": mixed_advantage.reshape(-1),
            "advantage": mixed_advantage.reshape(-1).detach(),
            "lambdas": blocks(lambdas).reshape(-1),
            "signals": {key: blocks(value).reshape(-1) for key, value in signals.items()},
            "current_logp": blocks(current_logp).reshape(-1),
            "supervised_tokens": usable_tokens,
            "supervision_units": usable_tokens,
        }
        if reference_logp is not None:
            result["reference_logp"] = blocks(reference_logp).reshape(-1)
        if reference_gap is not None:
            result["reference_gap"] = blocks(reference_gap).reshape(-1)
        return result

    if block_advantage_mode == "mean":
        unit_raw_advantage = raw_blocks.mean(dim=1)
        unit_advantage = weighted_raw_blocks.mean(dim=1)
    else:
        unit_raw_advantage = raw_blocks.sum(dim=1)
        unit_advantage = weighted_raw_blocks.sum(dim=1)
    result = {
        "weights": blocks(weights).mean(dim=1),
        "raw_advantage": unit_raw_advantage,
        "advantage": unit_advantage.detach(),
        "lambdas": lambda_blocks.mean(dim=1),
        "signals": {key: blocks(value).mean(dim=1) for key, value in signals.items()},
        "current_logp": blocks(current_logp).sum(dim=1),
        "supervised_tokens": usable_tokens,
        "supervision_units": supervision_units,
    }
    if reference_logp is not None:
        result["reference_logp"] = blocks(reference_logp).sum(dim=1)
    if reference_gap is not None:
        result["reference_gap"] = blocks(reference_gap).sum(dim=1)
    return result


def sample_rows(rows: list[dict[str, Any]], rng: random.Random, n: int) -> list[dict[str, Any]]:
    return [rows[rng.randrange(len(rows))] for _ in range(n)]


def save_checkpoint(
    out_dir: Path,
    step: int,
    student,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    args: argparse.Namespace,
) -> Path:
    ckpt = out_dir / f"checkpoint-step-{step:05d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(ckpt, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(ckpt)
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "sampler_rng_state": rng.getstate(),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "args": vars(args),
    }
    torch.save(state, ckpt / "training_state.pt")
    (ckpt / "training_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    (out_dir / "latest_checkpoint.txt").write_text(str(ckpt) + "\n", encoding="utf-8")
    return ckpt


def load_resume_state(path: str, device: torch.device) -> dict[str, Any]:
    state_path = Path(path) / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"missing full training state: {state_path}")
    return torch.load(state_path, map_location=device, weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-dir", required=True)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=[
            "standard",
            "topk_router",
            "adaptive_exopd",
            "routed_adaptive",
            "outcome_reweight",
            "outcome_topk_router",
            "anchored_standard",
            "anchored_topk_router",
            "ref_penalty_standard",
            "ref_penalty_topk_router",
        ],
        required=True,
    )
    parser.add_argument("--reference-dir", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--overlap-min", type=float, default=0.20)
    parser.add_argument("--margin-min", type=float, default=0.50)
    parser.add_argument("--entropy-max", type=float, default=2.0)
    parser.add_argument("--router-floor", type=float, default=0.10)
    parser.add_argument("--exopd-lambda", type=float, default=1.25)
    parser.add_argument("--correct-traj-weight", type=float, default=0.50)
    parser.add_argument("--wrong-traj-weight", type=float, default=1.50)
    parser.add_argument("--reference-penalty-beta", type=float, default=0.05)
    parser.add_argument(
        "--token-supervision-stride",
        type=int,
        default=1,
        help="Use only one completion token every N positions for OPD loss; 1 keeps dense token OPD.",
    )
    parser.add_argument(
        "--token-supervision-offset",
        type=int,
        default=0,
        help="Offset used with --token-supervision-stride.",
    )
    parser.add_argument(
        "--opd-block-size",
        type=int,
        default=1,
        help="Use non-overlapping sampled reverse-KL blocks of this many completion tokens; 1 keeps token OPD.",
    )
    parser.add_argument(
        "--opd-block-advantage-mode",
        choices=["sum", "mean", "mixed"],
        default="sum",
        help="How to turn token advantages into block supervision: sum is naive block OPD, mean normalizes by block length, mixed preserves token units with block-mean context.",
    )
    parser.add_argument(
        "--opd-block-mix-lambda",
        type=float,
        default=0.5,
        help="For --opd-block-advantage-mode mixed, interpolation weight for block mean advantage.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.token_supervision_stride < 1:
        raise ValueError("--token-supervision-stride must be >= 1")
    if args.token_supervision_offset < 0 or args.token_supervision_offset >= args.token_supervision_stride:
        raise ValueError("--token-supervision-offset must satisfy 0 <= offset < stride")
    if args.opd_block_size < 1:
        raise ValueError("--opd-block-size must be >= 1")
    if args.opd_block_size > 1 and (
        args.token_supervision_stride != 1 or args.token_supervision_offset != 0
    ):
        raise ValueError("--opd-block-size > 1 cannot be combined with token supervision stride/offset")
    if args.opd_block_mix_lambda < 0.0 or args.opd_block_mix_lambda > 1.0:
        raise ValueError("--opd-block-mix-lambda must satisfy 0 <= lambda <= 1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = load_jsonl(Path(args.data_jsonl))
    rng = random.Random(args.seed)
    resume_state = None
    load_student_dir = args.student_dir
    start_step = 0
    if args.resume_checkpoint:
        resume_state = load_resume_state(args.resume_checkpoint, device)
        load_student_dir = args.resume_checkpoint
        start_step = int(resume_state["step"])
        rng.setstate(resume_state["sampler_rng_state"])
        random.setstate(resume_state["python_random_state"])
        torch.set_rng_state(resume_state["torch_rng_state"].detach().cpu())
        torch.cuda.set_rng_state(resume_state["cuda_rng_state"].detach().cpu(), device=device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(load_student_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = AutoModelForCausalLM.from_pretrained(
        load_student_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    )
    student.train()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    reference = None
    if args.variant.startswith("anchored_") or args.variant.startswith("ref_penalty_"):
        reference_dir = args.reference_dir or args.student_dir
        reference = AutoModelForCausalLM.from_pretrained(
            reference_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": device},
        )
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])

    manifest = {
        "args": vars(args),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_rows": len(rows),
        "code": "clean-room independent script: scripts/clean_opd_train.py",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    metrics_path = out_dir / "train_metrics.jsonl"
    for step in range(start_step, args.max_steps):
        t0 = time.time()
        batch = sample_rows(rows, rng, args.prompts_per_step)
        optimizer.zero_grad(set_to_none=True)
        total_tokens = 0
        total_loss_units = 0
        total_loss = 0.0
        metric_lists: dict[str, list[float]] = defaultdict(list)

        for row in batch:
            prompt_text = render_prompt(tokenizer, row["prompt"], args.enable_thinking)
            enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_prompt_tokens,
            ).to(device)
            prompt_len = int(enc["input_ids"].shape[1])
            student.eval()
            with torch.no_grad():
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
            response_text = tokenizer.decode(gen[0, prompt_len : prompt_len + completion_len], skip_special_tokens=False)
            has_answer = bool(str(row.get("answer", "")).strip())
            trajectory_correct = is_correct(response_text, str(row.get("answer", ""))) if has_answer else False
            trajectory_weight = 1.0
            if args.variant.startswith("outcome_") and has_answer:
                trajectory_weight = args.correct_traj_weight if trajectory_correct else args.wrong_traj_weight
            sampled_ids = gen[:, 1:].squeeze(0)[prompt_len - 1 : prompt_len - 1 + completion_len]

            old_logp, student_top_ids, student_top_logp = token_logprobs_topk(student, gen, prompt_len, args.top_k)
            teacher_logp, teacher_top_ids, teacher_top_logp = token_logprobs_topk(teacher, gen, prompt_len, args.top_k)
            reference_logp = None
            if reference is not None:
                reference_logp = token_logprobs_topk(reference, gen, prompt_len, args.top_k)[0]
            signals = reliability_signals(
                sampled_ids,
                student_top_ids,
                student_top_logp,
                teacher_top_ids,
                teacher_top_logp,
                args.overlap_min,
                args.margin_min,
                args.entropy_max,
            )
            weights, lambdas = variant_weights(args.variant, signals["reliable"], args.router_floor, args.exopd_lambda)
            if args.variant.startswith("outcome_"):
                weights = weights * float(trajectory_weight)
            raw_advantage = (teacher_logp - old_logp).clamp(-20, 20)
            reference_gap = None
            if reference_logp is not None:
                reference_gap = (old_logp - reference_logp).clamp(-20, 20)
                if args.variant.startswith("anchored_"):
                    raw_advantage = (raw_advantage - reference_gap).clamp(-20, 20)

            student.train()
            current_logp = token_logprobs(student, gen, prompt_len)
            units = build_supervision_units(
                weights=weights,
                raw_advantage=raw_advantage,
                lambdas=lambdas,
                signals=signals,
                current_logp=current_logp,
                reference_logp=reference_logp,
                reference_gap=reference_gap,
                token_supervision_stride=args.token_supervision_stride,
                token_supervision_offset=args.token_supervision_offset,
                block_size=args.opd_block_size,
                block_advantage_mode=args.opd_block_advantage_mode,
                block_mix_lambda=args.opd_block_mix_lambda,
            )
            supervised_tokens = int(units["supervised_tokens"])
            supervision_units = int(units["supervision_units"])
            if supervision_units <= 0:
                continue
            weights = units["weights"]
            advantage = units["advantage"]
            raw_advantage = units["raw_advantage"]
            lambdas = units["lambdas"]
            signals = units["signals"]
            current_logp = units["current_logp"]
            reference_gap = units.get("reference_gap")
            reference_logp = units.get("reference_logp")
            denom = max(1, supervision_units)
            loss = -(weights.detach() * advantage * current_logp).sum() / denom
            reference_penalty = None
            current_reference_gap = None
            if args.variant.startswith("ref_penalty_"):
                if reference_logp is None:
                    raise RuntimeError("ref_penalty variants require a loaded reference model")
                current_reference_gap = (current_logp - reference_logp.detach()).clamp(-20, 20)
                reference_penalty = (weights.detach() * current_reference_gap.pow(2)).sum() / denom
                loss = loss + args.reference_penalty_beta * reference_penalty
            loss.backward()

            total_tokens += supervised_tokens
            total_loss_units += denom
            total_loss += float(loss.detach().cpu())
            metric_lists["completion_tokens"].append(float(completion_len))
            metric_lists["supervised_tokens"].append(float(supervised_tokens))
            metric_lists["supervision_units"].append(float(supervision_units))
            metric_lists["supervision_block_size"].append(float(args.opd_block_size))
            metric_lists["block_mix_lambda"].append(float(args.opd_block_mix_lambda))
            metric_lists["supervised_token_rate"].append(float(supervised_tokens) / max(1, float(completion_len)))
            metric_lists["supervision_unit_rate"].append(float(supervision_units) / max(1, float(completion_len)))
            metric_lists["advantage_mean"].append(float(advantage.mean().detach().cpu()))
            metric_lists["positive_advantage_rate"].append(float((advantage > 0).float().mean().detach().cpu()))
            metric_lists["raw_advantage_mean"].append(float(raw_advantage.mean().detach().cpu()))
            if reference_gap is not None:
                metric_lists["reference_gap_mean"].append(float(reference_gap.mean().detach().cpu()))
                metric_lists["positive_reference_gap_rate"].append(float((reference_gap > 0).float().mean().detach().cpu()))
            if reference_penalty is not None and current_reference_gap is not None:
                metric_lists["reference_penalty"].append(float(reference_penalty.detach().cpu()))
                metric_lists["current_reference_gap_mean"].append(float(current_reference_gap.mean().detach().cpu()))
                metric_lists["current_reference_gap_abs_mean"].append(float(current_reference_gap.abs().mean().detach().cpu()))
            metric_lists["overlap_mean"].append(float(signals["overlap"].mean().detach().cpu()))
            metric_lists["sampled_in_teacher_rate"].append(float(signals["sampled_in_teacher"].mean().detach().cpu()))
            metric_lists["teacher_margin_mean"].append(float(signals["teacher_margin"].mean().detach().cpu()))
            metric_lists["teacher_entropy_mean"].append(float(signals["teacher_entropy"].mean().detach().cpu()))
            metric_lists["router_keep_rate"].append(float(signals["reliable"].mean().detach().cpu()))
            metric_lists["lambda_mean"].append(float(lambdas.mean().detach().cpu()))
            metric_lists["weight_mean"].append(float(weights.mean().detach().cpu()))
            metric_lists["trajectory_correct"].append(float(trajectory_correct))
            metric_lists["trajectory_weight"].append(float(trajectory_weight))

        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        next_step = step + 1
        metrics = {
            "step": next_step,
            "variant": args.variant,
            "block_advantage_mode": args.opd_block_advantage_mode,
            "loss": total_loss,
            "tokens": total_tokens,
            "loss_units": total_loss_units,
            "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "sec": time.time() - t0,
            "lr": optimizer.param_groups[0]["lr"],
        }
        for key, values in metric_lists.items():
            metrics[key] = sum(values) / max(1, len(values))
        append_jsonl(metrics_path, metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

        if args.save_steps > 0 and next_step % args.save_steps == 0:
            save_checkpoint(out_dir, next_step, student, tokenizer, optimizer, rng, args)

    if args.save_steps <= 0 or args.max_steps % args.save_steps != 0:
        save_checkpoint(out_dir, args.max_steps, student, tokenizer, optimizer, rng, args)
    (out_dir / "done.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
