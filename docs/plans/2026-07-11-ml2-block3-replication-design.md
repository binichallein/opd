# ML2 Block3 Replication Design

## Objective

Replicate the `train` host's successful `block3_mean` experiment on `ml2`, while collecting the full Block10-era diagnostic surface and complete Step 50/100/200 evaluations. The run tests whether the small Block3 gain is observed on a second host; it does not claim multi-seed statistical replication.

## Fixed Experimental Contract

- Student: `Qwen3-1.7B-Base`.
- Teacher: `Qwen3-4B-Base-GRPO`.
- Training data: the same DAPO-Math-17K parquet with SHA-256 `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`.
- Algorithm: sampled block PPO with block size 3, mean block advantage, summed block current/old log-probability, and block-level PPO clipping.
- Seed: 21 for data, rollout, and environment.
- Training: 200 steps, 4 prompts per step, 8 rollouts per prompt, PPO mini-batch 32, actor PPO micro-batch/GPU 1, LR `2e-6`, maximum response length 16,384.
- Decoding: temperature 1.0 and top-p 0.9.
- Memory-engineering settings on `ml2`: reference log-probability micro-batch/GPU 1 and vLLM GPU memory utilization 0.6. These differ from the historical `train` run and must be recorded as a comparability caveat; they do not change the mathematical objective.

## Run Layout

Use a new dated run root under the existing TOS-backed repository:

```text
/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/
  runs/20260711_block3_replication_seed21_ml2/
    block3_mean/
      run_card.json
      command.sh
      env.txt
      artifact_hashes.sha256
      script_hashes.sha256
      started_at.txt
      finished_at.txt
      exit_code.txt
      train.pid
      logs/
      diagnostics/
      checkpoints/global_step_{50,100,200}/
      eval_step_{50,100,200}_n8/
```

All caches and temporary files must live under `/limx_embap/tos`, because the `ml2` root filesystem has only about 2 GiB free.

## Diagnostics

Save scalar diagnostics at Step 1 and every 5 steps, with full output-position stride 1 and Top-16 distributions:

- Student and teacher entropy.
- Signed and absolute entropy gap.
- Top-16 overlap ratio.
- Student and teacher overlap probability mass.
- Overlap-token advantage.
- SignFlipRate and WeightedSignFlipRate.
- LeakageMagnitude and NormalizedLeakage.
- Raw token and block advantage summaries.
- Post-update block log-ratio and block-ratio outside-clip fraction.
- PG loss, PPO clip fraction, pre-clipping gradient norm.
- Response length and truncation ratio.
- Rollout-vs-training-policy log-probability drift.

The position snapshots must be sufficient to regenerate entropy, overlap, sign-flip, leakage, and block-ratio heatmaps after training. Prompt-batch hashes must be recorded so shared-step data identity can be checked against future runs.

## Checkpoints And Evaluation

- Preserve complete checkpoints at Step 50, 100, and 200. Do not use automatic retention or delete earlier milestones.
- Each checkpoint must include model, optimizer, scheduler/extra state, data state, and all rank-specific state needed by VERL resume.
- Evaluate all three checkpoints on Math500, AIME24, AIME25, and AMC23.
- Use `n=8`, seeds 21-28, temperature 1.0, top-p 0.9, maximum 16,384 tokens, `enable_thinking=false`, and the VERL grader.
- Report per-task Avg@8, Pass@8, format-error count, and average response length, plus unweighted macro Avg@8 and Pass@8.

## Execution Gates

1. A two-step probe must complete with the same diagnostics and memory settings.
2. The probe must write a complete Step-1 checkpoint, resume from it, and advance to Step 2.
3. The formal run may start only if GPUs are idle, required assets and hashes exist, the TOS cache path is writable, and the probe has no CUDA OOM or non-finite diagnostic values.
4. Evaluation may start only after the corresponding complete checkpoint passes the audit.
5. The final report may claim completion only after all 41 diagnostic snapshots, all three checkpoints, and all six host/checkpoint evaluation summaries required for comparison are present and internally consistent.

## Analysis And Deliverables

- Generate scalar plots and position heatmaps using a Block3-capable analysis entry point rather than hard-coding Block10 labels.
- Compare `ml2` Block3 Step 200 with the historical `train` Block3 and token OPD summaries.
- Separate observations from causal claims. One seed per host can show repeatability across these two runs, not statistical robustness.
- Commit the launcher, tests, design/plan, curated machine-readable summary, plots, and an HTML report. Do not commit raw checkpoints, logs, or rollout JSONL files.

## Failure Handling

- Do not silently reduce sequence length, batch size, number of rollouts, or diagnostic coverage after a failure.
- If the probe OOMs, preserve the failed run and report the exact allocation site before proposing a new configuration.
- If the formal run exits unexpectedly, inspect checkpoint completeness and resume only through the verified VERL resume path.
- Treat the known post-training Ray DataLoader shutdown warning separately from CUDA OOM; success requires main exit code 0 and complete artifacts.
