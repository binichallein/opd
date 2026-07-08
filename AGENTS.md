# Agent Instructions

This repository is an internal OPD research workspace. Optimize for
reproducibility, data safety, and clear experiment lineage.

## Hard Rules

- Do not use user-trained or private checkpoints unless the user explicitly
  requests that exact model in the current task.
- Default student for current clean-room experiments: official `Qwen/Qwen3-0.6B`.
- Default teacher for current clean-room experiments: official `Qwen/Qwen3-4B`.
- Never commit checkpoints, model weights, raw datasets, processed datasets,
  full prediction JSONL files, logs, cache directories, SSH keys, tokens, or
  `.env` files.
- Do not put access keys, tokens, private keys, or passwords into committed
  files. Internal run paths may appear in result notes/configs when needed for
  lineage.
- Every real experiment should have a config, command, data manifest, run
  summary, and result note.
- Full eval means GSM8K 1319 examples and MATH500 500 examples unless a result
  note explicitly says it is a pilot or partial eval.

## Repository Map

- `scripts/clean_opd_train.py`: current OPD trainer and loss implementation.
- `scripts/eval_math_batched.py`: batched math evaluation.
- `scripts/regrade_predictions.py`: post-hoc regrading.
- `scripts/launch_*`: launch wrappers for remote experiments.
- `scripts/summarize_*`: compact experiment summaries.
- `tests/test_block_supervision.py`: unit tests for token/block supervision.
- `reports/`: human-readable HTML reports.
- `docs/results/`: concise experiment conclusions.
- `results/*.json`: small curated summaries only.

## Before Running Expensive Training

1. Confirm the exact model paths are official models unless told otherwise.
2. Confirm data file and manifest identity.
3. Run or inspect a resume gate.
4. Launch with `nohup` or another detached server-side mechanism.
5. Save complete training state: model, optimizer, scheduler, global step,
   sampler or consumed offset, RNG, args, and data identity.
6. Preserve checkpoints unless the user explicitly approves deletion.

## Before Reporting Completion

Run fresh verification. At minimum:

```bash
python -m py_compile scripts/clean_opd_train.py
python -m pytest tests/test_block_supervision.py -q
git status --short
```

If the task involved a remote run, also verify the run summary, output counts,
and that no training/eval process is left running unintentionally.
