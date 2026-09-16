# Agent Instructions

This repository is an internal OPD research workspace. Optimize for
reproducibility, data safety, and clear experiment lineage.

## Authorized Token Eval Then Block3 (2026-09-17)

- Latest user explicitly authorizes full evaluation of the completed non-thinking
  Qwen06 Token arm, followed by matched Block3 mean training. This overrides the
  historical eval/Block3 pause below, but NEVER restarts the old thinking queue.
- Read `docs/plans/2026-09-17-qwen06-token-eval-then-block3.md`. New ordered queue:
  `20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2`, controller
  `scripts/run_nonthinking_eval_block3.py`. Check live state before launching.
- Token source is the completed Sep13 v4 non-thinking run. Evaluate Steps200/100/50,
  official 0.6B Base and existing 4B GRPO teacher with pinned historical grader,
  full n8, all four benchmarks scored independently. No pooled headline score.
- Only after ALL five model evaluations pass raw/graded/coverage checks may the
  queue launch Block3's two-step resume probe and formal200. No weight warm-start
  from Token/probes. No new seed, sliding/random variant, SFT or private model.
- Block3 uses EXACT Token training runtime `0ce73aa42d7f734b9d34c379f474458bb6d48771`;
  controller has its own release. Keep explicit non-thinking, full rollout archives,
  original diagnostics and all50/100/200 complete checkpoints. Do not hot-edit
  either runtime or overwrite Token outputs. Merges/evals go to the NEW run root.
- Queue is launched: controller PID4130269, release
  `788a2dca21902fda25777fca1a6863780973a8c1`. Step200 full evaluation PID4130750
  reached four-GPU MATH500 generation; Block3 is still gated behind evaluations.
  Read `docs/results/2026-09-17-qwen06-eval-block3-startup.md` and LIVE queue state.
  Later documentation commits must not change either frozen runtime identity.
- User requested continuous supervision. Accepted per-benchmark scores and
  monitoring evidence are in `docs/results/2026-09-17-qwen06-evaluation-progress.md`.
  Read live state first; phase snapshots are not proof that subsequent jobs ran.
- Any eval failure or paired mismatch stops the queue. No auto retry/cleanup.
  Block3 full evaluation is NOT auto-started by this controller.

## Authorized Non-Thinking Token Run (2026-09-13)

- User now authorizes a NEW official Qwen3-0.6B-Base Token OPD run on ml2, with
  explicit thinking disable, training/evaluation prompt agreement, lossless
  training trajectories and all existing OPD diagnostics. Read
  `docs/plans/2026-09-13-qwen06-nonthinking-token.md` before acting.
- The active attempt is `20260913v4_qwen06_nonthinking_token_seed21_ml2`;
  controller `scripts/run_qwen06_nonthinking.py`. It gates training on real GPU
  prompt/output checks and a two-step training-state/rollout-retention resume test.
- Attempt v2 stopped in the GPU gate's CPU input construction (missing batch on a
  test stub), before any generation or training. Its runtime 8a0a2d3 and artifacts
  remain intact. v3 uses a real DataProto fixture; do not restart v2.
- v3 / 7724141 passed prompt checks but stopped before generation because its
  standalone vLLM gate forked after CUDA initialization. v4 explicitly uses spawn
  for that gate; no old runtime or failed output is overwritten.
- Keep original seed21, data, teacher, 200 steps and Step50/100/200 state retention.
  Start formal Token from the official Base, never from old/probe weights. New
  protocol also matches evaluation EOS stops and uses actual generation lengths
  for masks. Do not silently attribute protocol changes to Block3.
- The OLD Qwen06 queue remains stopped. Do not launch Block3 or revive old eval.
  This standalone controller trains Token and builds diagnostic figures only;
  full benchmark evaluation is not automatically resumed by this authorization.
- Inspect live state before launching; never duplicate a running controller.
- v4 controller PID3844168 uses frozen runtime
  `0ce73aa42d7f734b9d34c379f474458bb6d48771`. Read
  `docs/results/2026-09-13-qwen06-nonthinking-startup.md` and live queue state.
  Later documentation commits must not replace this runtime in audit commands.
- GPU gate and two-step four-rank resume/rollout audits passed. Formal Token
  PID3867901 started at 18:06:07 Beijing, and its first update/archive/diagnostic
  figures are verified. Training is ongoing, not complete. Both the GPU gate and
  actual first training batch produced no new think tags, but length truncation
  persists (formal Step1: 16/32). Do not claim non-thinking solved truncation.

## Non-Thinking Training Policy (2026-09-13)

- User requires all subsequent training to disable thinking mode. This overrides
  historical prompt defaults; it does not authorize restarting the paused queue.
- Explicitly pass `enable_thinking=False` to supported chat templates, including
  the active rollout path (`data.apply_chat_template_kwargs` in Revisiting OPD).
  An empty kwargs mapping is not an explicit disable. Remove environment/prompt
  instructions requiring reasoning inside `<think>...</think>` as well.
- Inspect the actual rendered prompt and token IDs in preflight, not just the
  run card. Qwen's tokenizer may express disabled thinking with an EMPTY,
  already-closed think block in the assistant prefix. Do not delete that control
  prefix just to make a string search find no think tags. This is not a guarantee
  that a Base model can never generate such tags or ordinary reasoning text.
- Preserve historic deployments, configs and trajectories unchanged. The paused
  Qwen06 runtime is not yet amended for this policy or full rollout retention;
  it must not be resumed blindly. Use a new versioned protocol and run identity.
- Apply the same protocol to Token and Block3 comparison arms. Do not compare a
  newly non-thinking Block3 arm against the historical thinking-unspecified,
  think-instructed Token arm as a controlled method-only ablation.
- Historical audit: ml2 1.7B Token/Block3 and current 0.6B Token training used
  empty template kwargs plus an environment instruction requiring think tags;
  their evaluation path explicitly passes false. The recent `no_think` diagnostic
  only removed an instruction; it did NOT set `enable_thinking=False`.
  See `docs/results/2026-09-13-token-truncation-comparison.md`.

## Active User Pause (2026-09-13)

- User stopped the Qwen06 evaluation queue to diagnose 0.6B truncation against
  1.7B. Controller PID3547500 and Step50 eval PID3822051 were terminated at
  2026-09-13 14:04 Beijing. The queue records `failed/error=143` and eval exit -15
  because of this deliberate stop, not a spontaneous training failure.
- Do not restart full evaluation, dispatch Block3, or relaunch the old controller
  until the user redirects from this diagnosis. All training checkpoints remain.
- Small matched inference-only diagnostics are authorized. Preserve raw tokens,
  special-token text and finish reasons; keep them separate from historical
  training trajectories and benchmark scores. Never call regenerated responses
  the original training rollouts.
- The bounded diagnosis is complete: 192 raw responses, all eight generation
  jobs exited 0. Read `docs/results/2026-09-13-qwen06-truncation-diagnosis.md`.
  The four GPUs were verified idle afterwards. Recommendations in that note are
  not launched experiments; the evaluation/Block3 pause still applies.

## Training Rollout Retention Policy (2026-09-13)

- User requires future training runs to retain their actual on-policy rollout
  trajectories. Scalar metrics, position heatmaps, and evaluation predictions
  are not substitutes for training trajectories.
- Save every training step and every sampled response, not only checkpoint or
  diagnostic steps. Retain run/attempt ID, step, prompt/sample/group identity,
  unmodified prompt and response token IDs, response length/mask, and decoded
  text that preserves special tokens. Link records to the run's sampling seeds,
  generation settings, tokenizer identity, and source revision.
- Preserve the generation engine's actual finish/stop reason when available,
  together with EOS/stop configuration. A length-at-cap indicator is a separate
  measurement, not proof of the engine's finish reason. Never invent missing
  reasons or reconstruct original trajectories by sampling a checkpoint again.
- Make retention part of preflight and the resume gate: verify readable records,
  step/sample coverage, special-token preservation, and no overwrite on resume.
  Logging must not consume sampling RNG or change loss, batches, or decoding.
- Keep raw trajectories in the experiment artifact store, never in Git. Preserve
  prior attempts and use distinct files/segments on resume; no silent pruning.
- This is a requirement for subsequent new launches, not evidence that the
  existing frozen Qwen06 queue already saves trajectories. Its completed Token
  run has no raw rollout dump. Do not hot-edit that runtime; any logging amendment
  to an already queued job must be explicit, versioned, and verified first.
- Historical truncation comparison and evidence limits are recorded in
  `docs/results/2026-09-13-token-truncation-comparison.md`.

## Benchmark Reporting Policy (2026-09-13)

- User requires independent scores for MATH500, AIME24, AIME25, and AMC23.
  Report each benchmark's Avg@8, Pass@8 and Block3-minus-Token differences at
  Step50/100/200. Never combine benchmarks into a total score or substitute a
  macro/pooled score or overall verdict for these separate results.
- Existing evaluation already retains task-specific graded files and `per_task`
  comparisons. Preserve historical raw artifacts and aggregate fields for lineage,
  but do not use their aggregate scores as new user-facing results.
- Frozen queue/report scripts still produce historical macro summaries. Before
  delivering a new final report, render the per-task results separately instead
  of publishing that old macro-first report unchanged. Do not interrupt training
  or regrade unchanged predictions merely to change presentation.
- A macro bootstrap interval is not a per-benchmark interval. Any task-specific
  uncertainty analysis must be calculated using that task's paired examples.

## Active Window Experiment (2026-09-11)

- New user-approved follow-on (2026-09-13): read
  `docs/plans/2026-09-13-qwen06-paired-validation.md`. Only official Qwen3-0.6B-Base
  with the existing public 4B GRPO teacher; Token OPD before original Block3 mean,
  identical data/seed/full eval. User subsequently removed the 48-hour time cap on
  2026-09-13; keep both runs at 200 steps with full eval. No 4B student
  or extra seeds. Wait for the current window queue to complete; never preempt it.
  Read `docs/results/2026-09-13-qwen06-pair-startup.md` before any action.
  Its queue is already running as PID3547500; do not launch a duplicate.
  On 2026-09-13 04:21 Beijing, the predecessor completed. Token resume gate
  passed, and formal Token training started at 04:47 as PID3594635, from Base
  with resume disabled. Check live state before acting; Block3 remains queued.
  Read the startup note's night-supervision section, including probe teardown
  warnings and the distinction between probe and formal diagnostics.
  Qwen06 runtime is frozen at
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
