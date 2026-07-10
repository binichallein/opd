"""Analysis helpers for Block10 collapse diagnostic runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def bin_position_statistics(
    statistics: dict[str, np.ndarray],
    bin_size: int,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute count-weighted position-bin means and coverage."""
    if bin_size < 1 or min_count < 1:
        raise ValueError("bin_size and min_count must be positive")
    sums = np.asarray(statistics["sum"], dtype=np.float64)
    counts = np.asarray(statistics["valid_count"], dtype=np.int64)
    if sums.shape != counts.shape or sums.ndim != 1:
        raise ValueError("position sums and counts must be one-dimensional and aligned")
    pad = (-sums.size) % bin_size
    if pad:
        sums = np.pad(sums, (0, pad))
        counts = np.pad(counts, (0, pad))
    binned_sum = sums.reshape(-1, bin_size).sum(axis=1)
    binned_count = counts.reshape(-1, bin_size).sum(axis=1)
    means = np.full(binned_sum.shape, np.nan, dtype=np.float64)
    valid = binned_count >= min_count
    means[valid] = binned_sum[valid] / binned_count[valid]
    return means, binned_count


def detect_sustained_onset(
    steps: np.ndarray,
    values: np.ndarray,
    direction: str,
    baseline_max_step: int = 30,
    mad_multiplier: float = 5.0,
    consecutive_points: int = 2,
) -> int | None:
    """Find the first sustained robust deviation from the early-step baseline."""
    steps = np.asarray(steps, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if steps.shape != values.shape or steps.ndim != 1:
        raise ValueError("steps and values must be aligned one-dimensional arrays")
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    baseline = values[(steps <= baseline_max_step) & np.isfinite(values)]
    if baseline.size < 3:
        return None
    median = float(np.median(baseline))
    mad = max(float(np.median(np.abs(baseline - median))), 1e-8)
    threshold = median + mad_multiplier * mad if direction == "up" else median - mad_multiplier * mad
    candidate = np.isfinite(values) & (steps > baseline_max_step)
    candidate &= values > threshold if direction == "up" else values < threshold
    for index in range(0, len(steps) - consecutive_points + 1):
        if np.all(candidate[index : index + consecutive_points]):
            return int(steps[index])
    return None


def classify_leading_mechanism(onsets: dict[str, int | None], tie_window: int = 5) -> str:
    """Apply the pre-registered temporal rules from the validation protocol."""
    mechanism_onsets = {
        "advantage_credit_leakage": _minimum_onset(onsets.get("sign_flip"), onsets.get("leakage")),
        "block_ratio_amplification": onsets.get("block_ratio"),
        "teacher_prefix_ood": onsets.get("teacher_ood"),
        "support_drift": onsets.get("support_drift"),
        "numerical_instability": onsets.get("numerical"),
    }
    available = {name: step for name, step in mechanism_onsets.items() if step is not None}
    if not available:
        return "undetermined"
    earliest = min(available.values())
    tied = [name for name, step in available.items() if step - earliest <= tie_window]
    if len(tied) != 1:
        return "mixed_mechanism"
    return tied[0]


def _minimum_onset(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def load_scalar_records(path: str | Path) -> list[dict[str, object]]:
    """Load scalar JSONL and let resumed steps replace earlier partial records."""
    records_by_step: dict[int, dict[str, object]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records_by_step[int(record["step"])] = record
    return [records_by_step[step] for step in sorted(records_by_step)]
