#!/usr/bin/env python3
"""Verify that a clean-room training checkpoint contains full state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    ckpt = Path(args.checkpoint)
    required = [
        "config.json",
        "tokenizer_config.json",
        "training_state.pt",
        "training_args.json",
    ]
    missing = [name for name in required if not (ckpt / name).exists()]
    state = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False) if not missing else {}
    state_required = [
        "step",
        "optimizer",
        "sampler_rng_state",
        "python_random_state",
        "torch_rng_state",
        "cuda_rng_state",
        "args",
    ]
    missing_state = [name for name in state_required if name not in state]
    summary = {
        "checkpoint": str(ckpt),
        "missing_files": missing,
        "missing_state_keys": missing_state,
        "step": state.get("step"),
        "ok": not missing and not missing_state,
    }
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
