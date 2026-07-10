# Model Policy

The current paper-aligned OPD experiments should use the same Qwen3 math OPD
model family as the Rethinking OPD / Blockwise Policy-Drift Gating line, unless
the user explicitly asks for a different model in the current task.

## Current Paper-Aligned Defaults

- Student: `Qwen3-1.7B-Base`
- Teacher: `Qwen3-4B-Base-GRPO`
- Training data: raw 1,791,700-row DAPO-Math-17K row-pool at
  `/mnt/data/cpfs/Yaleon/opd/data/math_opd_dapo17k_hf_full_eval4/train.parquet`
- Train paths on `train`:
  - Student: `/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-1.7B-Base`
  - Teacher: `/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-4B-Base-GRPO`

`external/revisiting_opd/` is the codebase substrate only. Its paper uses a
Qwen2.5/OpenThinker setup, but this repository's DAPO-Math-17K block experiment
should follow the Qwen3-1.7B / Qwen3-4B-GRPO setting above.

## Legacy Clean-Room Pilot Defaults

- Student: official `Qwen/Qwen3-0.6B`
- Teacher: official `Qwen/Qwen3-4B`

These were used for the first small clean-room pilot and should not be reported
as the paper-aligned DAPO-Math-17K setting.

## Do Not Use By Default

- User-trained checkpoints
- Private checkpoints
- Local checkpoints whose provenance is unclear
- Prior project checkpoints unrelated to the current OPD experiment

If a non-default model is used, record why it was used and where it came from
in the run note.
