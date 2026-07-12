# ML2 Token OPD Paired Validation Contract

## Objective

Test whether `block3_mean` improves over sampled-token OPD when both methods run
on the same ml2 host under the same immutable training runtime and evaluation
contract.

## Training Pair

- Runtime commit: `9b3b8b76bcdf02d0a4cfe4720cecb24bc7203977`.
- Student: `Qwen3-1.7B-Base`.
- Teacher: `Qwen3-4B-Base-GRPO`.
- Data: raw 1,791,700-row DAPO-Math-17K pool, SHA-256
  `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`.
- Seed 21, train batch 4, eight rollouts per prompt, 200 steps, LR `2e-6`,
  max response 16,384.
- Four A100 80GB GPUs, actor/ref/rollout-logprob micro-batches `1/1/4`, vLLM
  utilization `0.6`.
- Preserve complete Step 50/100/200 checkpoints without automatic deletion.
- Only objective difference: `token_opd` uses block size 1 and
  `block3_mean` uses block size 3 with mean token advantage.

## Evaluation Pair

- Checkpoints: Step 50, 100, and 200.
- Tasks: Math500, AIME24, AIME25, AMC23.
- `n=8`, rollout seeds 21-28, temperature 1.0, top-p 0.9, max tokens 16,384,
  thinking disabled.
- Each checkpoint must contain exactly 5,144 rollouts.
- Built-in VERL scores remain an immutable sensitivity view.
- The primary paired comparison uses the historical external grader, SHA-256
  `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`,
  because it covers the AMC23 answer format used by the historical baseline.

## Pairing And Uncertainty

- Require exact equality of `(task, example_id, rollout_id, seed)` keys between
  token and Block3 graded outputs.
- Compute per-prompt Avg@8 and Pass@8 differences.
- Compute task-level and equal-task macro differences.
- Estimate 95% confidence intervals with 10,000 task-stratified prompt bootstrap
  replicates using bootstrap seed 20260712.
- Report prompt win/tie/loss counts and all task deltas; do not report only the
  favorable aggregate.

## Decision Rule

- `single_seed_paired_validated`: both Step-200 macro Avg@8 and Pass@8 point
  deltas are positive and both paired 95% confidence intervals are strictly
  above zero.
- `directionally_supported`: both point deltas are positive, but at least one
  confidence interval includes zero.
- `not_supported`: either Step-200 macro point delta is zero or negative.

This decision concerns one paired training seed. It is not multi-seed evidence
and must not be generalized to training-run variance without additional seeds.
