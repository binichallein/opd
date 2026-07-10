"""Pure diagnostic calculations for sampled block OPD experiments."""

from __future__ import annotations

from collections.abc import Collection
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def preserve_module_training_mode(method):
    """Restore ``self.actor_module.training`` after an inference-style method."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        was_training = bool(self.actor_module.training)
        try:
            return method(self, *args, **kwargs)
        finally:
            self.actor_module.train(was_training)

    return wrapped


@torch.no_grad()
def prompt_batch_sha256(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_length: int,
) -> str:
    """Hash the order-invariant multiset of prompt token sequences in a rollout batch."""
    if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
        raise ValueError("input_ids and attention_mask must be aligned rank-two tensors")
    if response_length < 1 or response_length >= input_ids.size(1):
        raise ValueError("response_length must leave a nonempty prompt segment")
    prompt_ids = input_ids[:, :-response_length].detach().cpu()
    prompt_mask = attention_mask[:, :-response_length].bool().detach().cpu()
    row_digests = []
    for row_ids, row_mask in zip(prompt_ids, prompt_mask):
        tokens = row_ids[row_mask].to(torch.int64).numpy().astype("<i8", copy=False)
        row_digests.append(hashlib.sha256(tokens.tobytes()).digest())
    return hashlib.sha256(b"".join(sorted(row_digests))).hexdigest()


def _pad_last_dim(tensor: torch.Tensor, multiple: int, value: float = 0.0) -> torch.Tensor:
    pad_size = (-tensor.size(-1)) % multiple
    if pad_size == 0:
        return tensor
    pad_shape = (*tensor.shape[:-1], pad_size)
    return torch.cat((tensor, tensor.new_full(pad_shape, value)), dim=-1)


@torch.no_grad()
def compute_block_credit_diagnostics(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    block_size: int,
    sign_epsilon: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Compare token advantages with the mean advantage shared by each block."""
    if advantages.shape != response_mask.shape:
        raise ValueError("advantages and response_mask must have identical shapes")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    mask = response_mask.bool()
    mask_float = mask.to(dtype=advantages.dtype)
    valid_advantages = torch.where(mask, advantages, torch.zeros_like(advantages))
    original_length = advantages.size(-1)

    advantage_blocks = _pad_last_dim(valid_advantages, block_size).reshape(
        advantages.size(0), -1, block_size
    )
    mask_blocks = _pad_last_dim(mask_float, block_size).reshape(
        advantages.size(0), -1, block_size
    )
    block_counts = mask_blocks.sum(dim=-1).clamp(min=1.0)
    block_means = advantage_blocks.sum(dim=-1) / block_counts
    block_per_token = (
        block_means.unsqueeze(-1).expand_as(advantage_blocks).reshape(advantages.size(0), -1)
    )[:, :original_length]
    block_per_token = torch.where(mask, block_per_token, torch.zeros_like(block_per_token))

    eligible = mask & (advantages.abs() > sign_epsilon) & (block_per_token.abs() > sign_epsilon)
    sign_flip = eligible & (torch.sign(advantages) != torch.sign(block_per_token))
    eligible_count = eligible.sum()
    sign_flip_count = sign_flip.sum()
    sign_flip_rate = sign_flip_count.to(advantages.dtype) / eligible_count.clamp(min=1)

    eligible_weights = torch.where(eligible, advantages.abs(), torch.zeros_like(advantages))
    flip_weights = torch.where(sign_flip, advantages.abs(), torch.zeros_like(advantages))
    weighted_sign_flip_rate = flip_weights.sum() / eligible_weights.sum().clamp(min=sign_epsilon)

    leakage = torch.where(mask, (block_per_token - advantages).abs(), torch.zeros_like(advantages))
    valid_count = mask.sum()
    leakage_magnitude = leakage.sum() / valid_count.clamp(min=1)
    raw_abs_sum = torch.where(mask, advantages.abs(), torch.zeros_like(advantages)).sum()
    normalized_leakage = leakage.sum() / raw_abs_sum.clamp(min=sign_epsilon)

    return {
        "block_advantage_per_token": block_per_token,
        "eligible_mask": eligible,
        "sign_flip_mask": sign_flip,
        "leakage_per_token": leakage,
        "eligible_count": eligible_count,
        "sign_flip_count": sign_flip_count,
        "sign_flip_rate": sign_flip_rate,
        "weighted_sign_flip_rate": weighted_sign_flip_rate,
        "leakage_magnitude": leakage_magnitude,
        "normalized_leakage": normalized_leakage,
    }


def _masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe_values = torch.where(mask, values, torch.zeros_like(values))
    return safe_values.sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def compute_block_ratio_diagnostics(
    old_log_prob: torch.Tensor,
    current_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    block_size: int,
    cliprange_low: float,
    cliprange_high: float,
    overflow_log_threshold: float = 80.0,
) -> dict[str, torch.Tensor]:
    """Summarize the block-level PPO ratio without changing policy-loss tensors."""
    if old_log_prob.shape != current_log_prob.shape or old_log_prob.shape != response_mask.shape:
        raise ValueError("log-probability tensors and response_mask must have identical shapes")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    mask = response_mask.bool()
    mask_float = mask.to(dtype=old_log_prob.dtype)
    log_ratio_per_token = torch.where(
        mask,
        current_log_prob - old_log_prob,
        torch.zeros_like(old_log_prob),
    )
    log_ratio_blocks = _pad_last_dim(log_ratio_per_token, block_size).reshape(
        old_log_prob.size(0), -1, block_size
    )
    mask_blocks = _pad_last_dim(mask_float, block_size).reshape(
        old_log_prob.size(0), -1, block_size
    )
    block_valid = mask_blocks.sum(dim=-1) > 0
    block_log_ratio = log_ratio_blocks.sum(dim=-1)
    finite = torch.isfinite(block_log_ratio) & block_valid
    nonfinite_count = (block_valid & ~torch.isfinite(block_log_ratio)).sum()
    overflow_count = (finite & (block_log_ratio > overflow_log_threshold)).sum()
    underflow_count = (finite & (block_log_ratio < -overflow_log_threshold)).sum()
    safe_log_ratio = torch.where(finite, block_log_ratio, torch.zeros_like(block_log_ratio))
    block_ratio = torch.exp(safe_log_ratio.clamp(max=overflow_log_threshold))
    clipped = finite & (
        (block_ratio < 1.0 - cliprange_low) | (block_ratio > 1.0 + cliprange_high)
    )
    clip_fraction = clipped.sum().to(block_ratio.dtype) / finite.sum().clamp(min=1)

    valid_abs_log_ratio = safe_log_ratio.abs()[finite]
    valid_ratio = block_ratio[finite]
    zero = block_ratio.new_zeros(())
    log_ratio_abs_mean = valid_abs_log_ratio.mean() if valid_abs_log_ratio.numel() else zero
    log_ratio_abs_p95 = (
        torch.quantile(valid_abs_log_ratio.float(), 0.95).to(block_ratio.dtype)
        if valid_abs_log_ratio.numel()
        else zero
    )
    log_ratio_abs_max = valid_abs_log_ratio.max() if valid_abs_log_ratio.numel() else zero
    ratio_p95 = (
        torch.quantile(valid_ratio.float(), 0.95).to(block_ratio.dtype)
        if valid_ratio.numel()
        else zero
    )
    ratio_max = valid_ratio.max() if valid_ratio.numel() else zero

    return {
        "block_log_ratio": block_log_ratio,
        "block_ratio": block_ratio,
        "block_valid_mask": block_valid,
        "clip_fraction": clip_fraction,
        "overflow_count": overflow_count,
        "underflow_count": underflow_count,
        "nonfinite_count": nonfinite_count,
        "log_ratio_abs_mean": log_ratio_abs_mean,
        "log_ratio_abs_p95": log_ratio_abs_p95,
        "log_ratio_abs_max": log_ratio_abs_max,
        "ratio_p95": ratio_p95,
        "ratio_max": ratio_max,
    }


@torch.no_grad()
def compute_entropy_diagnostics(
    student_entropy: torch.Tensor,
    teacher_entropy: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute signed and absolute entropy gaps on student-generated states."""
    if student_entropy.shape != teacher_entropy.shape or student_entropy.shape != response_mask.shape:
        raise ValueError("entropy tensors and response_mask must have identical shapes")
    mask = response_mask.bool()
    signed_gap = torch.where(mask, teacher_entropy - student_entropy, torch.zeros_like(student_entropy))
    absolute_gap = signed_gap.abs()
    return {
        "signed_gap": signed_gap,
        "absolute_gap": absolute_gap,
        "student_mean": _masked_scalar_mean(student_entropy, mask),
        "teacher_mean": _masked_scalar_mean(teacher_entropy, mask),
        "signed_gap_mean": _masked_scalar_mean(signed_gap, mask),
        "absolute_gap_mean": _masked_scalar_mean(absolute_gap, mask),
    }


@torch.no_grad()
def compute_topk_alignment_diagnostics(
    student_topk_ids: torch.Tensor,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute Rethinking OPD alignment metrics on student/teacher top-k sets."""
    if student_topk_ids.shape != student_topk_log_probs.shape:
        raise ValueError("student top-k ids and log-probabilities must have identical shapes")
    if teacher_topk_ids.shape != teacher_topk_log_probs.shape:
        raise ValueError("teacher top-k ids and log-probabilities must have identical shapes")
    if student_topk_ids.shape[:-1] != teacher_topk_ids.shape[:-1]:
        raise ValueError("student and teacher top-k tensors must share batch and position dimensions")
    if student_topk_ids.size(-1) != teacher_topk_ids.size(-1):
        raise ValueError("student and teacher must use the same top-k size")
    if student_topk_ids.shape[:-1] != response_mask.shape:
        raise ValueError("response_mask must match the top-k batch and position dimensions")

    matches = student_topk_ids.unsqueeze(-1) == teacher_topk_ids.unsqueeze(-2)
    student_overlap = matches.any(dim=-1)
    teacher_match_index = matches.to(torch.int64).argmax(dim=-1)
    teacher_log_probs_on_student = teacher_topk_log_probs.gather(-1, teacher_match_index)

    overlap_count = student_overlap.sum(dim=-1)
    position_valid = response_mask.bool()
    overlap_valid = position_valid & (overlap_count > 0)
    overlap_ratio = overlap_count.to(student_topk_log_probs.dtype) / student_topk_ids.size(-1)

    student_prob = student_topk_log_probs.exp()
    teacher_prob_on_student = teacher_log_probs_on_student.exp()
    overlap_float = student_overlap.to(student_prob.dtype)
    student_mass = (student_prob * overlap_float).sum(dim=-1)
    teacher_mass = (teacher_prob_on_student * overlap_float).sum(dim=-1)

    log_student_mass = student_mass.clamp_min(torch.finfo(student_prob.dtype).tiny).log()
    log_teacher_mass = teacher_mass.clamp_min(torch.finfo(student_prob.dtype).tiny).log()
    student_bar = torch.where(
        student_overlap,
        torch.exp(student_topk_log_probs - log_student_mass.unsqueeze(-1)),
        torch.zeros_like(student_topk_log_probs),
    )
    log_ratio_bar = (
        teacher_log_probs_on_student
        - log_teacher_mass.unsqueeze(-1)
        - student_topk_log_probs
        + log_student_mass.unsqueeze(-1)
    )
    advantage_sum = torch.where(
        student_overlap,
        student_bar * log_ratio_bar,
        torch.zeros_like(student_bar),
    ).sum(dim=-1)
    overlap_advantage = advantage_sum / overlap_count.clamp(min=1)

    zero = torch.zeros_like(overlap_ratio)
    return {
        "overlap_ratio": torch.where(position_valid, overlap_ratio, zero),
        "student_overlap_mass": torch.where(position_valid, student_mass, zero),
        "teacher_overlap_mass": torch.where(position_valid, teacher_mass, zero),
        "overlap_token_advantage": torch.where(overlap_valid, overlap_advantage, zero),
        "overlap_valid_mask": overlap_valid,
    }


@torch.no_grad()
def compute_position_statistics(values: torch.Tensor, mask: torch.Tensor) -> dict[str, np.ndarray]:
    """Return per-position sum, squared sum, and valid count as CPU arrays."""
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    mask_bool = mask.bool()
    safe_values = torch.where(mask_bool, values, torch.zeros_like(values)).to(torch.float64)
    return {
        "sum": safe_values.sum(dim=0).cpu().numpy(),
        "squared_sum": safe_values.square().sum(dim=0).cpu().numpy(),
        "valid_count": mask_bool.sum(dim=0).to(torch.int64).cpu().numpy(),
    }


@torch.no_grad()
def compute_masked_summary(values: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Summarize finite masked values while reporting excluded non-finite entries."""
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    selected = values[mask.bool()]
    finite_values = selected[torch.isfinite(selected)].float()
    zero = values.new_zeros((), dtype=torch.float32)
    if finite_values.numel():
        mean = finite_values.mean()
        std = finite_values.std(unbiased=False)
        p95 = torch.quantile(finite_values, 0.95)
        maximum = finite_values.max()
    else:
        mean = std = p95 = maximum = zero
    return {
        "count": values.new_tensor(selected.numel(), dtype=torch.int64),
        "finite_count": values.new_tensor(finite_values.numel(), dtype=torch.int64),
        "nonfinite_count": values.new_tensor(
            selected.numel() - finite_values.numel(), dtype=torch.int64
        ),
        "mean": mean,
        "std": std,
        "p95": p95,
        "max": maximum,
    }


def scatter_position_statistics(
    statistics: dict[str, np.ndarray],
    positions: np.ndarray,
    output_length: int,
) -> dict[str, np.ndarray]:
    """Scatter sampled-position statistics into the full response coordinate system."""
    required = {"sum", "squared_sum", "valid_count"}
    if set(statistics) != required:
        raise ValueError(f"position statistics must contain exactly {sorted(required)}")
    positions = np.asarray(positions, dtype=np.int64)
    if np.any(positions < 0) or np.any(positions >= output_length):
        raise ValueError("sampled positions fall outside the response length")
    if any(np.asarray(statistics[key]).shape != positions.shape for key in required):
        raise ValueError("sampled statistics and positions must have identical shapes")

    result = {
        "sum": np.zeros(output_length, dtype=np.float64),
        "squared_sum": np.zeros(output_length, dtype=np.float64),
        "valid_count": np.zeros(output_length, dtype=np.int64),
    }
    for key in required:
        result[key][positions] = np.asarray(statistics[key], dtype=result[key].dtype)
    return result


def should_save_checkpoint(
    global_step: int,
    is_last_step: bool,
    milestone_steps: Collection[int],
) -> bool:
    """Return whether a full checkpoint is required at this step."""
    return bool(is_last_step or global_step in milestone_steps)


def should_run_diagnostics(global_step: int, enabled: bool, interval: int) -> bool:
    """Run the first diagnostic snapshot and then follow the configured interval."""
    if interval < 1:
        raise ValueError("diagnostic interval must be positive")
    return bool(enabled and (global_step == 1 or global_step % interval == 0))


def should_stop_after_step(global_step: int, stop_after_step: int) -> bool:
    """Support a probe-only early stop without changing the scheduler horizon."""
    return bool(stop_after_step > 0 and global_step == stop_after_step)


def parse_milestone_steps(value: str | Collection[int]) -> set[int]:
    """Parse and validate the full-state checkpoint milestone set."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        try:
            milestones = {int(item.strip()) for item in stripped.split(",")}
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint milestone list: {value}") from exc
    else:
        milestones = {int(item) for item in value}
    if any(step <= 0 for step in milestones):
        raise ValueError("checkpoint milestones must be positive")
    return milestones


def merge_position_stat_shards(shards: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Merge worker-local position sums without averaging worker means."""
    required = {"sum", "squared_sum", "valid_count"}
    if set(shards) != required:
        raise ValueError(f"position-stat shards must contain exactly {sorted(required)}")
    return {
        "sum": shards["sum"].sum(dim=0).detach().to(torch.float64).cpu().numpy(),
        "squared_sum": shards["squared_sum"].sum(dim=0).detach().to(torch.float64).cpu().numpy(),
        "valid_count": shards["valid_count"].sum(dim=0).detach().to(torch.int64).cpu().numpy(),
    }


def save_diagnostic_snapshot(
    output_dir: str | Path,
    step: int,
    scalars: dict[str, float],
    position_stats: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, int | float | str | bool],
) -> Path:
    """Atomically save position arrays and append a compact scalar record."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_path = output_path / f"step_{step:05d}.npz"
    arrays: dict[str, np.ndarray] = {"step": np.asarray(step, dtype=np.int64)}
    arrays.update({key: np.asarray(value) for key, value in metadata.items()})
    for metric_name, stats in position_stats.items():
        for statistic_name in ("sum", "squared_sum", "valid_count"):
            arrays[f"{metric_name}__{statistic_name}"] = np.asarray(stats[statistic_name])

    with tempfile.NamedTemporaryFile(
        dir=output_path,
        prefix=f".{final_path.name}.",
        suffix=".tmp.npz",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    scalar_record = {"step": int(step), **scalars, **metadata}
    scalar_path = output_path / "scalars.jsonl"
    with scalar_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(scalar_record, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return final_path


def load_scalar_records(path: str | Path) -> list[dict[str, object]]:
    """Load scalar JSONL, keeping the latest record if a resumed run repeats a step."""
    records_by_step: dict[int, dict[str, object]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records_by_step[int(record["step"])] = record
    return [records_by_step[step] for step in sorted(records_by_step)]
