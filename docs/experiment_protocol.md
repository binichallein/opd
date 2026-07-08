# Experiment Protocol

This document defines the minimum standard for OPD experiments in this
repository.

## Required Experiment Inputs

- Experiment config file under `configs/experiments/`
- Exact command or launch script
- Student and teacher model identifiers
- Data file path and manifest/hash
- Seed, rollout settings, training settings, and eval settings
- Expected output directory

## Required Run Directory Contents

A real run directory should be self-describing:

```text
runs/<run-id>/
  run_card.json
  command.sh
  config.resolved.yaml
  git_commit.txt
  git_diff.patch
  env.txt
  data_manifest.json
  logs/
  checkpoints/
  summaries/
```

The run directory itself should not be committed to git. Commit only a small
summary in `docs/results/` and, when useful, compact JSON tables in `results/`.

## Checkpoint Contract

Every checkpoint intended for resume must include:

- Model weights, tokenizer, and config
- Optimizer state
- Scheduler state
- Global step
- Consumed sample offset or deterministic sampler state
- Python, torch, and CUDA RNG state
- Run args
- Data identity

Do not treat HF weights alone as a resumable training checkpoint.

## Launch Policy

Long-running remote jobs must be started server-side with `nohup`, a scheduler,
or an equivalent detached mechanism. Do not rely on a local SSH session or a
foreground pipe to keep training alive.

## Evaluation Policy

Unless a result is explicitly labeled as a pilot:

- GSM8K eval uses all 1319 examples.
- MATH500 eval uses all 500 examples.
- Final reported scores should come from post-hoc regrade summaries, not only
  online rough grading.
