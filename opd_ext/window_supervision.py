"""Pure phase-partitioned, mean-advantage joint-ratio OPD supervision.

Every contiguous valid region restarts its phase coordinates. Phase r keeps an
initial partial window of length r, then windows of length k and a partial tail.
Sliding mode averages complete phase losses, not advantages before clipping.
"""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real

import numpy as np
import torch


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _phases(block_size, window_mode, offset):
    block_size = _integer(block_size, "block_size", 1)
    offset = _integer(offset, "offset")
    if offset >= block_size:
        raise ValueError("offset must satisfy 0 <= offset < block_size")
    if window_mode not in ("random", "sliding"):
        raise ValueError("window_mode must be 'random' or 'sliding'")
    return range(block_size) if window_mode == "sliding" else (offset,)


def _finite_scalar(value, name, minimum=0, maximum=math.inf):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return float(value)


def _clip_ranges(low, high):
    return _finite_scalar(low, "cliprange_low", maximum=1), _finite_scalar(high, "cliprange_high")


def _require_finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite on all valid tokens/windows")


def _validate_tensors(response_mask, **values):
    if not isinstance(response_mask, torch.Tensor) or response_mask.ndim != 2:
        raise ValueError("response_mask must be a rank-two tensor")
    if response_mask.is_complex() or not ((response_mask == 0) | (response_mask == 1)).all():
        raise ValueError("response_mask must be binary")
    dtype = torch.float32
    for name, value in values.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.shape != response_mask.shape
        ):
            raise ValueError(f"{name} and response_mask must have matching rank-two shapes")
        if not value.is_floating_point() or value.device != response_mask.device:
            raise ValueError(f"{name} must be floating point and on the response_mask device")
        dtype = torch.promote_types(dtype, value.dtype)
    return response_mask.detach().bool(), dtype


def _masked(value, mask, dtype, *, detach=True):
    if detach:
        value = value.detach()
    return torch.where(mask, value.to(dtype), 0)


def _window_ids(mask, block_size, offset):
    """Use each window's first token as its slot in a fixed B x T allocation."""
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    previous = torch.nn.functional.pad(mask, (1, 0), value=False)[:, :-1]
    starts = torch.where(mask & ~previous, positions, 0).cummax(dim=1).values
    local = positions - starts
    window_start = torch.where(
        local < offset,
        0,
        offset + torch.div(local - offset, block_size, rounding_mode="floor") * block_size,
    )
    return torch.where(mask, starts + window_start, 0)


def _window_sum(value, ids):
    return torch.zeros_like(value).scatter_add(1, ids, value)


def _phase_loss(delta, advantages, mask, block_size, offset, low, high, clipc):
    ids = _window_ids(mask, block_size, offset)
    counts = _window_sum(mask.to(delta.dtype), ids)
    log_ratio = _window_sum(delta, ids)
    advantage_sum = _window_sum(advantages, ids)
    _require_finite(log_ratio, "window log ratio")
    _require_finite(advantage_sum, "window advantage sum")
    mean_advantage = advantage_sum / counts.clamp(min=1)
    ratio = log_ratio.exp()
    _require_finite(ratio, "window ratio")
    raw = -mean_advantage * ratio
    clipped = -mean_advantage * ratio.clamp(1 - low, 1 + high)
    dual = -mean_advantage * clipc
    _require_finite(raw, "unclipped window loss")
    _require_finite(clipped, "clipped window loss")
    _require_finite(dual, "dual-clipped window loss")
    upper = torch.maximum(raw, clipped)
    losses = torch.where(mean_advantage < 0, torch.minimum(upper, dual), upper)
    denominator = mask.sum().clamp(min=1)
    terms = (
        losses,
        (clipped > raw).to(delta.dtype),
        -log_ratio,
        ((upper > dual) & (mean_advantage < 0)).to(delta.dtype),
    )
    result = tuple((term * counts).sum() / denominator for term in terms)
    for value in result:
        _require_finite(value, "reduced loss/metric")
    return result


def compute_window_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    block_size=3,
    window_mode="random",
    offset=0,
    cliprange=0.2,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Return upstream's (loss, clipfrac, ppo_kl, dualclipfrac) scalar tuple.

    Each window contributes its valid length times dual-clip PPO of its mean
    advantage and joint ratio, divided by this reduction unit's valid token count.
    Only current log-probabilities receive gradients. Half inputs use FP32;
    FP64 inputs stay FP64. Empty masks yield differentiable zero. Valid nonfinite
    inputs/intermediates raise ValueError; nonfinite padding is removed first.

    Numerical change from historical block aggregation: sum(current - old)
    avoids cancellation in sum(current) - sum(old). This is mathematically
    identical but intentionally not bitwise identical for large log-probabilities.
    """
    phases = _phases(block_size, window_mode, offset)
    if loss_agg_mode != "token-mean":
        raise ValueError("window policy loss supports only token-mean aggregation")
    low, high = _clip_ranges(
        cliprange if cliprange_low is None else cliprange_low,
        cliprange if cliprange_high is None else cliprange_high,
    )
    clipc = _finite_scalar(clip_ratio_c, "clip_ratio_c", minimum=1)
    if clipc <= 1:
        raise ValueError("clip_ratio_c must be greater than 1")
    mask, dtype = _validate_tensors(
        response_mask, old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages
    )
    old = _masked(old_log_prob, mask, dtype)
    current = _masked(log_prob, mask, dtype, detach=False)
    adv = _masked(advantages, mask, dtype)
    for name, value in (("old_log_prob", old), ("log_prob", current), ("advantages", adv)):
        _require_finite(value, name)
    delta = current - old
    _require_finite(delta, "token log ratio")
    results = [_phase_loss(delta, adv, mask, block_size, r, low, high, clipc) for r in phases]
    averaged = tuple(torch.stack([result[i] for result in results]).mean() for i in range(4))
    for value in averaged:
        _require_finite(value, "phase-averaged loss/metric")
    return averaged


@torch.no_grad()
def window_credit_diagnostics(
    advantages,
    response_mask,
    block_size=3,
    window_mode="random",
    offset=0,
    sign_epsilon=1e-4,
):
    """Historical shared-mean credit metrics plus the true unit-ratio coefficient.

    ``block_advantage_per_token`` averages expanded per-phase window MEANS.
    ``effective_token_coefficient`` averages expanded per-phase window SUMS:
    at current == old it equals -N * d(loss)/d(current_log_prob), not the
    historical shared-mean view. Away from unit ratios/clipping it is only the
    credit coefficient, not the actual loss derivative. Coverage counts include
    every phase (one for random, k for sliding) and are zero on padding.
    """
    phases = _phases(block_size, window_mode, offset)
    epsilon = _finite_scalar(sign_epsilon, "sign_epsilon")
    if epsilon == 0:
        raise ValueError("sign_epsilon must be positive")
    mask, dtype = _validate_tensors(response_mask, advantages=advantages)
    adv = _masked(advantages, mask, dtype)
    _require_finite(adv, "advantages")
    means, sums = torch.zeros_like(adv), torch.zeros_like(adv)
    for phase in phases:
        ids = _window_ids(mask, block_size, phase)
        counts = _window_sum(mask.to(dtype), ids)
        total = _window_sum(adv, ids)
        _require_finite(total, "window advantage sum")
        means += (total / counts.clamp(min=1)).gather(1, ids) / len(phases)
        sums += total.gather(1, ids) / len(phases)
    means = torch.where(mask, means, 0)
    sums = torch.where(mask, sums, 0)
    eligible = mask & (adv.abs() > epsilon) & (means.abs() > epsilon)
    flips = eligible & (adv.sign() != means.sign())
    eligible_weights = torch.where(eligible, adv.abs(), 0)
    flip_weights = torch.where(flips, adv.abs(), 0)
    leakage = torch.where(mask, (means - adv).abs(), 0)
    result = {
        "block_advantage_per_token": means,
        "eligible_mask": eligible,
        "sign_flip_mask": flips,
        "leakage_per_token": leakage,
        "eligible_count": eligible.sum(),
        "sign_flip_count": flips.sum(),
        "sign_flip_rate": flips.sum().to(dtype) / eligible.sum().clamp(min=1),
        "weighted_sign_flip_rate": flip_weights.sum() / eligible_weights.sum().clamp(min=epsilon),
        "leakage_magnitude": leakage.sum() / mask.sum().clamp(min=1),
        "normalized_leakage": leakage.sum() / adv.abs().sum().clamp(min=epsilon),
        "effective_token_coefficient": sums,
        "coverage_per_token": mask.to(torch.int64) * len(phases),
    }
    for name, value in result.items():
        _require_finite(value, name)
    return result


@torch.no_grad()
def window_ratio_diagnostics(
    old_log_prob,
    current_log_prob,
    response_mask,
    block_size=3,
    window_mode="random",
    offset=0,
    cliprange_low=0.2,
    cliprange_high=0.2,
):
    """Pool active windows for historical scalar ratio statistics.

    Nonfinite windows are counted, excluded from scalar summaries, and mark all
    their tokens nonfinite. Token log-ratio magnitude is the mean of expanded
    per-phase absolute window log-ratios, zero unless ALL covering windows are
    finite. Token clipping is ANY finite covering window outside the clip range.
    Coverage counts all active windows, including nonfinite ones. As historically,
    overflow/underflow use +/-80 and reported ratios cap the exponent at 80;
    these diagnostic caps never affect policy loss. FP64 quantiles stay FP64.
    """
    phases = _phases(block_size, window_mode, offset)
    low, high = _clip_ranges(cliprange_low, cliprange_high)
    mask, dtype = _validate_tensors(
        response_mask, old_log_prob=old_log_prob, current_log_prob=current_log_prob
    )
    delta = _masked(current_log_prob, mask, dtype) - _masked(old_log_prob, mask, dtype)
    token_abs = torch.zeros_like(delta)
    token_finite = mask.clone()
    token_outside = torch.zeros_like(mask)
    all_logs = []
    all_finite = []
    for phase in phases:
        ids = _window_ids(mask, block_size, phase)
        counts = _window_sum(mask.to(dtype), ids)
        log_ratio = _window_sum(delta, ids)
        valid = counts > 0
        finite = valid & torch.isfinite(log_ratio)
        safe = torch.where(finite, log_ratio, 0)
        ratio = safe.clamp(max=80).exp()
        outside = finite & ((ratio < 1 - low) | (ratio > 1 + high))
        token_abs += safe.abs().gather(1, ids) / len(phases)
        token_finite &= finite.gather(1, ids)
        token_outside |= mask & outside.gather(1, ids)
        all_logs.append(log_ratio[valid])
        all_finite.append(finite[valid])
    logs = torch.cat(all_logs)
    finite = torch.cat(all_finite)
    finite_logs = logs[finite]
    abs_logs = finite_logs.abs()
    ratios = finite_logs.clamp(max=80).exp()
    outside = (ratios < 1 - low) | (ratios > 1 + high)
    zero = delta.new_zeros(())
    return {
        "clip_fraction": outside.sum().to(dtype) / finite.sum().clamp(min=1),
        "overflow_count": (finite_logs > 80).sum(),
        "underflow_count": (finite_logs < -80).sum(),
        "nonfinite_count": (~finite).sum(),
        "log_ratio_abs_mean": abs_logs.mean() if abs_logs.numel() else zero,
        "log_ratio_abs_p95": torch.quantile(abs_logs, 0.95) if abs_logs.numel() else zero,
        "log_ratio_abs_max": abs_logs.max() if abs_logs.numel() else zero,
        "ratio_p95": torch.quantile(ratios, 0.95) if ratios.numel() else zero,
        "ratio_max": ratios.max() if ratios.numel() else zero,
        "token_log_ratio_abs": torch.where(token_finite, token_abs, 0),
        "token_finite_mask": token_finite,
        "token_outside_clip": token_outside,
        "coverage_per_token": mask.to(torch.int64) * len(phases),
    }


def window_phase_diagnostics(
    old_log_prob,
    current_log_prob,
    advantages,
    response_mask,
    block_size=3,
    clip_low=0.2,
    clip_high=0.2,
    clipc=3.0,
):
    """Scalar statistics of d(loss)/d(current_log_prob), NOT parameter gradients.

    Private detached leaves permit use inside no_grad/inference_mode without
    changing any caller graph, .grad buffer, model mode, or RNG. Disagreement is
    mean_r ||g_r - mean(g)||^2. Identity errors are absolute loss difference and
    the L2 gradient difference against an independently built sliding loss.
    Pairwise cosine uses a zero sentinel when undefined; the matching scalar
    ``phase_{r}_{s}_cosine_valid`` flag distinguishes it from orthogonality.
    """
    _phases(block_size, "sliding", 0)
    _, dtype = _validate_tensors(
        response_mask,
        old_log_prob=old_log_prob,
        current_log_prob=current_log_prob,
        advantages=advantages,
    )
    with torch.inference_mode(False), torch.enable_grad():
        old = old_log_prob.detach().to(dtype).clone()
        current = current_log_prob.detach().to(dtype).clone().requires_grad_()
        adv = advantages.detach().to(dtype).clone()
        mask = response_mask.detach().clone()
        kwargs = dict(
            block_size=block_size,
            cliprange_low=clip_low,
            cliprange_high=clip_high,
            clip_ratio_c=clipc,
        )
        losses, gradients = [], []
        for phase in range(block_size):
            loss = compute_window_policy_loss(old, current, adv, mask, offset=phase, **kwargs)[0]
            gradients.append(torch.autograd.grad(loss, current)[0])
            losses.append(loss.detach())
        sliding = compute_window_policy_loss(
            old, current, adv, mask, window_mode="sliding", **kwargs
        )[0]
        sliding_gradient = torch.autograd.grad(sliding, current)[0]
    with torch.no_grad():
        gradients = torch.stack(gradients)
        mean = gradients.mean(dim=0)
        norms = gradients.flatten(1).norm(dim=1)
        result = {
            "phase_disagreement_squared_norm": (gradients - mean).square().flatten(1).sum(1).mean(),
            "mean_phase_gradient_norm": norms.mean(),
            "mean_gradient_norm": mean.norm(),
            "loss_identity_error": (sliding.detach() - torch.stack(losses).mean()).abs(),
            "gradient_identity_error": (sliding_gradient - mean).norm(),
        }
        for phase in range(block_size):
            result[f"phase_{phase}_loss"] = losses[phase]
            result[f"phase_{phase}_gradient_norm"] = norms[phase]
            for other in range(phase + 1, block_size):
                denominator = norms[phase] * norms[other]
                product = (gradients[phase] * gradients[other]).sum()
                valid = (denominator > 0) & torch.isfinite(denominator) & torch.isfinite(product)
                result[f"phase_{phase}_{other}_cosine_valid"] = valid
                result[f"phase_{phase}_{other}_cosine"] = torch.where(
                    valid, product / denominator.clamp(min=torch.finfo(dtype).tiny), 0
                )
        return result


def reconcile_diagnostic_resume(output_dir, step):
    """Archive uncheckpointed diagnostics before replaying their optimizer steps."""
    directory = Path(output_dir)
    step = _integer(step, "step", 1)
    rewrites = []
    for name in ("window_steps.jsonl", "scalars.jsonl"):
        path = directory / name
        if not path.exists():
            if name == "window_steps.jsonl":
                raise FileNotFoundError(f"Missing window resume ledger: {path}")
            continue
        lines = path.read_text().splitlines(keepends=True)
        records = [json.loads(line) for line in lines]
        if name == "window_steps.jsonl":
            prefix = [r["step"] for r in records if r["step"] <= step]
            if prefix != list(range(1, step + 1)):
                raise ValueError("Window resume ledger has an incomplete or duplicated saved prefix")
        retained = [line for line, record in zip(lines, records) if record["step"] <= step]
        if len(retained) != len(lines):
            rewrites.append((path, "".join(retained)))
    future = [path for path in directory.glob("step_*.npz") if int(path.stem.split("_")[-1]) > step]
    if not rewrites and not future:
        return None
    history = directory / "resume_history"
    history.mkdir(exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix=f"after_step_{step}_", dir=history))
    for path, content in rewrites:
        shutil.copy2(path, archive / path.name)
        temporary = path.with_suffix(".resume.tmp")
        temporary.write_text(content)
        os.replace(temporary, path)
    for path in future:
        shutil.move(path, archive / path.name)
    return archive


class OffsetSchedule:
    """One private PCG64 offset draw per strictly sequential optimizer step.

    The first call is next(1); repeated/skipped steps are errors. The driver owns
    broadcasting/reusing that offset across ranks and micro-batches. JSON-ready
    state has exactly block_size, seed, last_step, current_offset, rng_state.
    """

    def __init__(self, block_size, seed):
        self.block_size = _integer(block_size, "block_size", 1)
        self.seed = _integer(seed, "seed")
        self.last_step = 0
        self.current_offset = None
        self._rng = np.random.Generator(np.random.PCG64(self.seed))

    def next(self, step):
        step = _integer(step, "step", 1)
        if step != self.last_step + 1:
            raise ValueError(f"step must be last_step + 1 ({self.last_step + 1}), got {step}")
        offset = int(self._rng.integers(self.block_size))
        self.last_step, self.current_offset = step, offset
        return offset

    def state_dict(self):
        return {
            "block_size": self.block_size,
            "seed": self.seed,
            "last_step": self.last_step,
            "current_offset": self.current_offset,
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        required = {"block_size", "seed", "last_step", "current_offset", "rng_state"}
        if not isinstance(state, Mapping) or not required.issubset(state):
            raise ValueError(f"offset state must contain {sorted(required)}")
        block_size = _integer(state["block_size"], "block_size", 1)
        seed = _integer(state["seed"], "seed")
        if block_size != self.block_size or seed != self.seed:
            raise ValueError("offset state block_size and seed must match this schedule exactly")
        last_step = _integer(state["last_step"], "last_step")
        offset = state["current_offset"]
        if last_step == 0:
            if offset is not None:
                raise ValueError("initial current_offset must be None")
        else:
            offset = _integer(offset, "current_offset")
            if offset >= block_size:
                raise ValueError("current_offset must be less than block_size")
        # Validate on a fresh generator so a malformed checkpoint cannot partly load.
        rng = np.random.Generator(np.random.PCG64(seed))
        try:
            rng.bit_generator.state = deepcopy(state["rng_state"])
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise ValueError("rng_state must be a valid PCG64 state") from exc
        self.last_step, self.current_offset, self._rng = last_step, offset, rng
