#!/usr/bin/env python3
"""Audit offline JSON window-resume evidence without Torch or training imports.

The standalone CLI writes window_acceptance.json, leaving the base acceptance
unchanged. audit_window_control.py combines this check with audit_block10_run.py
for the existing control's preflight gates; queues must run both audits too.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

VARIANT_MODES = {"random3": "random", "sliding3": "sliding"}
WINDOW_SEED = 910021


def _read_object(path: Path, issues: list[str]) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        issues.append(f"{path.name}: {exc}")
        return {}


def _is_uint(value: object, upper: int) -> bool:
    return type(value) is int and 0 <= value < upper


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_pcg64_state(value: object) -> bool:
    if not isinstance(value, dict) or value.get("bit_generator") != "PCG64":
        return False
    state = value.get("state")
    return (
        isinstance(state, dict)
        and _is_uint(state.get("state"), 2**128)
        and _is_uint(state.get("inc"), 2**128)
        and state["inc"] % 2 == 1
        and _is_uint(value.get("has_uint32"), 2)
        and _is_uint(value.get("uinteger"), 2**32)
    )


def window_issues(
    run_dir: Path,
    variant: str,
    checkpoint_steps: list[int],
    expected_last_step: int | None = None,
) -> list[str]:
    """Require actual state and every completed step, not only a variant label."""
    if variant not in VARIANT_MODES:
        raise ValueError(f"unsupported window variant: {variant}")
    if not checkpoint_steps or any(type(step) is not int or step <= 0 for step in checkpoint_steps):
        raise ValueError("checkpoint steps must be positive integers")
    mode = VARIANT_MODES[variant]
    issues: list[str] = []
    card = _read_object(run_dir / "run_card.json", issues)
    expected = {
        "variant": variant,
        "seed": 21,
        "opd_block_size": 3,
        "opd_block_advantage_mode": "mean",
        "opd_window_mode": mode,
        "opd_window_seed": WINDOW_SEED,
        "ppo_epochs": 1,
    }
    for key, value in expected.items():
        if card.get(key) != value or type(card.get(key)) is not type(value):
            issues.append(f"run_card {key}: expected {value!r}, got {card.get(key)!r}")

    digest = card.get("window_supervision_sha256")
    if not _is_sha256(digest):
        issues.append("run_card window_supervision_sha256: missing or invalid SHA256")
    hash_path = run_dir / "script_hashes.sha256"
    try:
        entries = [line.split(maxsplit=1) for line in hash_path.read_text().splitlines()]
        module_hashes = [
            parts[0]
            for parts in entries
            if len(parts) == 2 and parts[1].lstrip("*").endswith("/opd_ext/window_supervision.py")
        ]
        if module_hashes != [digest]:
            issues.append(
                "script_hashes.sha256: window_supervision.py hash missing or inconsistent"
            )
    except OSError as exc:
        issues.append(f"script_hashes.sha256: {exc}")

    log_path = run_dir / "diagnostics/window_steps.jsonl"
    records = []
    try:
        for line_number, line in enumerate(log_path.read_text().splitlines(), 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
                records.append(record)
            except ValueError as exc:
                issues.append(f"window_steps.jsonl line {line_number}: {exc}")
    except OSError as exc:
        issues.append(f"window_steps.jsonl: {exc}")

    last_step = max(checkpoint_steps) if expected_last_step is None else expected_last_step
    if type(last_step) is not int or last_step < max(checkpoint_steps):
        issues.append("expected last step must be at least the maximum checkpoint step")
        last_step = max(checkpoint_steps)
    actual_steps = [record.get("step") for record in records]
    if any(type(step) is not int for step in actual_steps) or actual_steps != list(
        range(1, last_step + 1)
    ):
        issues.append(f"window_steps.jsonl: expected exactly ordered steps 1..{last_step}")
    by_step = {}
    for record in records:
        step = record.get("step")
        if type(step) is int:
            by_step[step] = record
        if record.get("window_mode") != mode:
            issues.append(f"window step {step}: window_mode must be {mode}")
        offset = record.get("window_offset")
        if not _is_uint(offset, 3) or (mode == "sliding" and offset != 0):
            issues.append(f"window step {step}: invalid window_offset {offset!r}")
        for key in ("prompt_batch_sha256", "prompt_schedule_sha256"):
            if not _is_sha256(record.get(key)):
                issues.append(f"window step {step}: missing or invalid {key}")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            issues.append(f"window step {step}: metrics must be a nonempty JSON object")
        elif any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in metrics.values()
        ):
            issues.append(f"window step {step}: metrics must contain only finite numbers")

    for step in checkpoint_steps:
        path = run_dir / "checkpoints" / f"global_step_{step}" / "window_state.json"
        state_issues: list[str] = []
        state = _read_object(path, state_issues)
        for key, expected_value in (
            ("mode", mode),
            ("block_size", 3),
            ("seed", WINDOW_SEED),
            ("last_step", step),
        ):
            if state.get(key) != expected_value or type(state.get(key)) is not type(expected_value):
                state_issues.append(f"{key}: expected {expected_value!r}, got {state.get(key)!r}")
        offset = state.get("current_offset")
        if not _is_uint(offset, 3) or (mode == "sliding" and offset != 0):
            state_issues.append(f"invalid current_offset {offset!r}")
        if offset != by_step.get(step, {}).get("window_offset"):
            state_issues.append("current_offset differs from the checkpoint step's log record")
        if "rng_state" not in state:
            state_issues.append("missing rng_state")
        elif mode == "random" and not _valid_pcg64_state(state["rng_state"]):
            state_issues.append("rng_state must be a complete PCG64 state")
        elif mode == "sliding" and state["rng_state"] is not None:
            state_issues.append("rng_state must be null for sliding mode")
        issues.extend(f"checkpoint {step}: {issue}" for issue in state_issues)
    return issues


def _atomic_write_json(path: Path, value: dict) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(*, include_legacy: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANT_MODES, required=True)
    parser.add_argument("--checkpoint-steps", required=True)
    parser.set_defaults(expected_last_step=None)
    if include_legacy:
        args, _ = parser.parse_known_args()
    else:
        parser.add_argument("--expected-last-step", type=int)
        args = parser.parse_args()
    try:
        steps = sorted({int(part) for part in args.checkpoint_steps.split(",")})
        if not steps or steps[0] <= 0:
            raise ValueError("checkpoint steps must be positive")
    except ValueError as exc:
        parser.error(str(exc))

    if include_legacy:
        # Preserve every historical check and argument, including resume and eval gates.
        legacy = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("audit_block10_run.py")), *sys.argv[1:]],
            check=False,
        )
    issues = window_issues(args.run_dir, args.variant, steps, args.expected_last_step)
    window_result = {
        "passed": not issues,
        "issues": list(issues),
        "run_dir": str(args.run_dir),
        "variant": args.variant,
        "mode": VARIANT_MODES[args.variant],
        "seed": WINDOW_SEED,
        "checkpoint_steps": steps,
        "last_step": max(steps) if args.expected_last_step is None else args.expected_last_step,
    }
    _atomic_write_json(args.run_dir / "window_acceptance.json", window_result)
    result = window_result
    if include_legacy:
        acceptance_path = args.run_dir / "acceptance.json"
        result = _read_object(acceptance_path, issues)
        if legacy.returncode:
            issues.append(f"existing audit failed with exit code {legacy.returncode}")
        if result.get("passed") is not True:
            issues.append("existing audit did not report passed=true")
        result["issues"] = [*result.get("issues", []), *issues]
        result["passed"] = not result["issues"]
        result["window_audit"] = window_result
        _atomic_write_json(acceptance_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
