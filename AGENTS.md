# Agent Instructions

This repository is an internal OPD research workspace. Optimize for
reproducibility, data safety, and clear experiment lineage.

## Active Window Experiment (2026-09-11)

- New user-approved follow-on (2026-09-13): read
  `docs/plans/2026-09-13-qwen06-paired-validation.md`. Only official Qwen3-0.6B-Base
  with the existing public 4B GRPO teacher; Token OPD before original Block3 mean,
  identical data/seed/full eval. User subsequently removed the 48-hour time cap on
  2026-09-13; keep both runs at 200 steps with full eval. No 4B student
  or extra seeds. Wait for the current window queue to complete; never preempt it.
  Read `docs/results/2026-09-13-qwen06-pair-startup.md` before any action.
  Its queue is already running as PID3547500, waiting for the old queue; do not
  launch a duplicate. Qwen06 runtime is frozen at
  `ec0a7a950540d7f4753c08b18bc26c544f6a7cde` (no time cap). The earlier waiting
  controller PID3544990 was retired before any GPU job; its evidence is archived.
  Later docs commits do not replace the runtime.

- Read `docs/plans/2026-09-11-sliding-window-opd-validation-v2-seed21.md`
  and `docs/results/2026-09-11-window-seed21-startup.md` before acting.
- This approved experiment uses only `ml2`; never connect to `train`.
- Only `random3` and `sliding3`, training seed21, 200 steps each. No automatic
  additional seeds, token/fixed baselines, or model pairs.
- Its explicit model/data/eval protocol supersedes the historical clean-room
  defaults below: student Qwen3-1.7B-Base, teacher Qwen3-4B-Base-GRPO, the recorded DAPO pool,
  all four math tasks with n=8 at Step50/100/200. Do not substitute other assets.
- Runtime is frozen at `fbad852a638de18e20d571a061b2be0437942a38`; pass that value
  as `SOURCE_COMMIT` to window control commands after documentation-only commits.
- Inspect live queue state before any launch. Do not duplicate a queued run or
  present resume-gate measurements as formal training/evaluation results.
- For the reviewed JSONL failure, read `docs/results/2026-09-12-window-eval-recovery.md`.
  Recovery uses a separate committed analysis release; never overwrite the frozen
  training runtime, original failed job, or completed raw/graded predictions.

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
