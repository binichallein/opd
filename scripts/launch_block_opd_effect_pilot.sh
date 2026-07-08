#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WS=${WS:-$ROOT}
CODE_ROOT=${CODE_ROOT:-$ROOT}
RUN_ROOT=${RUN_ROOT:-$WS/runs/block_opd_effect_$(date +%Y%m%d_%H%M%S)}
DETACH=${DETACH:-1}

if [[ "$DETACH" == "1" && "${BLOCK_OPD_ATTACHED:-0}" != "1" ]]; then
  mkdir -p "$RUN_ROOT/logs"
  BLOCK_OPD_ATTACHED=1 RUN_ROOT="$RUN_ROOT" DETACH=0 nohup bash "$0" \
    > "$RUN_ROOT/logs/launch.log" 2>&1 &
  echo $! > "$RUN_ROOT/launch.pid"
  echo "block_opd_pilot_started pid=$(cat "$RUN_ROOT/launch.pid") run_root=$RUN_ROOT log=$RUN_ROOT/logs/launch.log"
  exit 0
fi

cd "$CODE_ROOT"
mkdir -p "$RUN_ROOT/logs"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$WS/.venv/bin/python" ]]; then
    PYTHON="$WS/.venv/bin/python"
  else
    PYTHON=python
  fi
fi
GPU=${GPU:-0}
STUDENT_DIR=${STUDENT_DIR:-$WS/models/official/Qwen__Qwen3-0.6B}
TEACHER_DIR=${TEACHER_DIR:-$WS/models/official/Qwen__Qwen3-4B}
TRAIN_JSONL=${TRAIN_JSONL:-$WS/data/public_math/train_prompts.jsonl}
GSM8K_JSONL=${GSM8K_JSONL:-$WS/data/public_math/eval_gsm8k.jsonl}
MATH500_JSONL=${MATH500_JSONL:-$WS/data/public_math/eval_math500.jsonl}
MAX_STEPS=${MAX_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-$MAX_STEPS}
PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-2}
TRAIN_MAX_NEW_TOKENS=${TRAIN_MAX_NEW_TOKENS:-128}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-256}
LEARNING_RATE=${LEARNING_RATE:-5e-7}
GRAD_CLIP=${GRAD_CLIP:-1.0}
TOP_K=${TOP_K:-20}
SEED=${SEED:-7}
EVAL_LIMIT=${EVAL_LIMIT:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}

if [[ "$PYTHON" == */* ]]; then
  if [[ ! -x "$PYTHON" ]]; then
    echo "missing executable python: $PYTHON" >&2
    exit 2
  fi
elif ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "python command not found: $PYTHON" >&2
  exit 2
fi

required_paths=(
  "$CODE_ROOT/scripts/clean_opd_train.py"
  "$CODE_ROOT/scripts/eval_math_batched.py"
  "$CODE_ROOT/scripts/summarize_block_opd_effect.py"
  "$STUDENT_DIR"
  "$TEACHER_DIR"
  "$TRAIN_JSONL"
  "$GSM8K_JSONL"
  "$MATH500_JSONL"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 2
  fi
done

export HF_HOME=${HF_HOME:-$WS/models/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}

run_train() {
  local name=$1
  local block_size=$2
  local out_dir="$RUN_ROOT/$name"
  mkdir -p "$out_dir"
  cat > "$out_dir/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$CODE_ROOT"
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$CODE_ROOT/scripts/clean_opd_train.py" \\
  --student-dir "$STUDENT_DIR" \\
  --teacher-dir "$TEACHER_DIR" \\
  --data-jsonl "$TRAIN_JSONL" \\
  --output-dir "$out_dir" \\
  --variant standard --max-steps $MAX_STEPS --save-steps $SAVE_STEPS \\
  --prompts-per-step $PROMPTS_PER_STEP --max-new-tokens $TRAIN_MAX_NEW_TOKENS \\
  --learning-rate $LEARNING_RATE --grad-clip $GRAD_CLIP --top-k $TOP_K \\
  --opd-block-size $block_size --seed $SEED --device cuda:0
EOF
  chmod +x "$out_dir/command.sh"
  bash "$out_dir/command.sh" > "$out_dir/train.log" 2>&1
}

run_eval() {
  local name=$1
  local ckpt="$RUN_ROOT/$name/checkpoint-step-$(printf "%05d" "$MAX_STEPS")"
  local out_dir="$RUN_ROOT/$name"
  if [[ ! -d "$ckpt" ]]; then
    ckpt=$(cat "$RUN_ROOT/$name/latest_checkpoint.txt")
  fi
  cat > "$out_dir/eval_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$CODE_ROOT"
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$CODE_ROOT/scripts/eval_math_batched.py" \\
  --model-dir "$ckpt" \\
  --data-jsonl "$GSM8K_JSONL" \\
  --output-jsonl "$out_dir/gsm8k_preds.jsonl" \\
  --summary-json "$out_dir/gsm8k_summary.json" \\
  --limit $EVAL_LIMIT --batch-size $EVAL_BATCH_SIZE --max-new-tokens $EVAL_MAX_NEW_TOKENS \\
  --seed $SEED --device cuda:0
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$CODE_ROOT/scripts/eval_math_batched.py" \\
  --model-dir "$ckpt" \\
  --data-jsonl "$MATH500_JSONL" \\
  --output-jsonl "$out_dir/math500_preds.jsonl" \\
  --summary-json "$out_dir/math500_summary.json" \\
  --limit $EVAL_LIMIT --batch-size $EVAL_BATCH_SIZE --max-new-tokens $EVAL_MAX_NEW_TOKENS \\
  --seed $SEED --device cuda:0
"$PYTHON" "$CODE_ROOT/scripts/summarize_block_opd_effect.py" --run-root "$RUN_ROOT" --output-json "$RUN_ROOT/compact_summary.json"
EOF
  chmod +x "$out_dir/eval_command.sh"
  bash "$out_dir/eval_command.sh" > "$out_dir/eval.log" 2>&1
}

printf '%s\n' "run_root=$RUN_ROOT" "started_at=$(date -Is)" > "$RUN_ROOT/run_info.txt"

run_train token_opd 1
run_eval token_opd
run_train block_k2 2
run_eval block_k2
run_train block_k3 3
run_eval block_k3

"$PYTHON" "$CODE_ROOT/scripts/summarize_block_opd_effect.py" --run-root "$RUN_ROOT" --output-json "$RUN_ROOT/compact_summary.json"
date -Is > "$RUN_ROOT/done.txt"
echo "block_opd_pilot_done run_root=$RUN_ROOT summary=$RUN_ROOT/compact_summary.json"
