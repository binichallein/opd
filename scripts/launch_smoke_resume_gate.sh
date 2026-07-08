#!/usr/bin/env bash
set -euo pipefail

WS=${WS:-/mnt/data/cpfs/Yaleon/opd_cleanroom_20260707}
VARIANT=${VARIANT:-standard}
GPU=${GPU:-0}
RUN=${RUN:-$WS/runs/smoke/${VARIANT}_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$RUN"
source "$WS/.venv/bin/activate"
export HF_HOME="$WS/models/hf_home"
export HF_HUB_CACHE="$WS/models/hf_home/hub"

COMMON=(
  "$WS/scripts/clean_opd_train.py"
  --student-dir "$WS/models/official/Qwen__Qwen3-0.6B"
  --teacher-dir "$WS/models/official/Qwen__Qwen3-4B"
  --data-jsonl "$WS/data/public_math/train_prompts.jsonl"
  --variant "$VARIANT"
  --prompts-per-step 1
  --max-new-tokens 64
  --save-steps 1
  --learning-rate 1e-6
  --top-k 10
  --device cuda:0
)

CUDA_VISIBLE_DEVICES="$GPU" python "${COMMON[@]}" --output-dir "$RUN/phase1" --max-steps 1
python "$WS/scripts/check_resume_checkpoint.py" "$RUN/phase1/checkpoint-step-00001"
CUDA_VISIBLE_DEVICES="$GPU" python "${COMMON[@]}" --output-dir "$RUN/phase2" --max-steps 2 --resume-checkpoint "$RUN/phase1/checkpoint-step-00001"
python "$WS/scripts/check_resume_checkpoint.py" "$RUN/phase2/checkpoint-step-00002"

echo "$RUN" > "$WS/runs/smoke/latest_${VARIANT}.txt"
echo "smoke_resume_gate_ok run=$RUN"
