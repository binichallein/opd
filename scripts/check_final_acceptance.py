#!/usr/bin/env python3
"""Classify a paired-validation run as pending, complete, or failed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STEPS = (50, 100, 200)


def acceptance_state(run: Path) -> tuple[str, str]:
    if not run.is_dir():
        return "failed", f"run directory is missing: {run}"
    exit_paths = [run / "exit_code.txt"] + [
        run / f"eval_step_{step}_n8" / "exit_code.txt" for step in STEPS
    ]
    for path in exit_paths:
        if not path.is_file():
            continue
        try:
            code = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            return "failed", f"invalid exit code: {path}"
        if code != 0:
            return "failed", f"nonzero exit code {code}: {path}"

    acceptance_path = run / "acceptance.json"
    if not acceptance_path.is_file():
        return "pending", "acceptance.json is missing"
    try:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return "failed", f"invalid acceptance.json: {error}"
    if not isinstance(acceptance, dict):
        return "failed", "acceptance schema must be a JSON object"
    if acceptance.get("passed") is not True:
        return "failed", "acceptance passed=false"
    if acceptance.get("issues"):
        return "failed", f"acceptance issues: {acceptance['issues']}"
    required = set(STEPS)
    checkpoint_steps = acceptance.get("checkpoint_steps")
    eval_steps = acceptance.get("eval_steps")
    if not isinstance(checkpoint_steps, list) or not isinstance(eval_steps, list):
        return "failed", "acceptance step schema must contain JSON arrays"
    if any(type(step) is not int for step in (*checkpoint_steps, *eval_steps)):
        return "failed", "acceptance step schema must contain integers"
    accepted_checkpoints = set(checkpoint_steps)
    accepted_evals = set(eval_steps)
    if not required.issubset(accepted_checkpoints):
        return "pending", "final checkpoint acceptance is incomplete"
    if not required.issubset(accepted_evals):
        return "pending", "final eval acceptance is incomplete"
    missing_exits = [str(path) for path in exit_paths if not path.is_file()]
    if missing_exits:
        return "pending", f"required exit code is missing: {missing_exits}"
    return "complete", "all final artifacts passed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        state, reason = acceptance_state(args.run_dir)
    except Exception as error:  # Fail closed on unexpected filesystem/schema errors.
        print(
            json.dumps(
                {"state": "runtime_error", "reason": repr(error)},
                ensure_ascii=False,
            )
        )
        raise SystemExit(3) from error
    print(json.dumps({"state": state, "reason": reason}, ensure_ascii=False))
    raise SystemExit({"complete": 0, "pending": 1, "failed": 2}[state])


if __name__ == "__main__":
    main()
