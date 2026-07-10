#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-train}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data/cpfs/Yaleon/opd}"
RUN_ROOT="${RUN_ROOT:-${REMOTE_ROOT}/runs/20260708_block3_dapo17k_paper_qwen3}"
VARIANT="${VARIANT:-token_opd}"
STEP="${STEP:-200}"
VENV="${VENV:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/venv}"
HF_HOME_DIR="${HF_HOME_DIR:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/hf_home}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/tmp/opd_block3_dapo17k_eval}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${REMOTE_ROOT}/data/math_opd_dapo17k_hf_full_eval4/eval_jsonl}"
LENGTH_TOKENIZER_PATH="${LENGTH_TOKENIZER_PATH:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-1.7B-Base}"
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
GPUS="${GPUS:-0,1,2,3}"
TASKS="${TASKS:-math500 aime24 aime25 amc23}"

RUN_DIR="${RUN_ROOT}/${VARIANT}"
CKPT_DIR="${RUN_DIR}/checkpoints/global_step_${STEP}"
ACTOR_DIR="${CKPT_DIR}/actor"
MODEL_DIR="${ACTOR_DIR}/huggingface"
EVAL_DIR="${RUN_DIR}/eval_step_${STEP}_n${N}"
LOG_DIR="${EVAL_DIR}/logs"

ssh "${REMOTE}" "set -euo pipefail
test -x '${VENV}/bin/python'
test -d '${ACTOR_DIR}'
test -f '${EVAL_DATA_DIR}/math500.jsonl'
test -f '${EVAL_DATA_DIR}/aime24.jsonl'
test -f '${EVAL_DATA_DIR}/aime25.jsonl'
test -f '${EVAL_DATA_DIR}/amc23.jsonl'
mkdir -p '${EVAL_DIR}' '${LOG_DIR}' \
  '${LOCAL_CACHE_ROOT}/tmp' '${LOCAL_CACHE_ROOT}/vllm_cache' \
  '${LOCAL_CACHE_ROOT}/torchinductor' '${LOCAL_CACHE_ROOT}/triton' \
  '${LOCAL_CACHE_ROOT}/cuda_cache' '${LOCAL_CACHE_ROOT}/outlines'
cat > '${EVAL_DIR}/eval_card.json' <<JSON
{
  \"variant\": \"${VARIANT}\",
  \"step\": ${STEP},
  \"checkpoint_dir\": \"${CKPT_DIR}\",
  \"actor_dir\": \"${ACTOR_DIR}\",
  \"model_dir\": \"${MODEL_DIR}\",
  \"eval_data_dir\": \"${EVAL_DATA_DIR}\",
  \"n\": ${N},
  \"temperature\": ${TEMPERATURE},
  \"top_p\": ${TOP_P},
  \"max_tokens\": ${MAX_TOKENS},
  \"gpus\": \"${GPUS}\",
  \"tasks\": \"${TASKS}\"
}
JSON
cat > '${EVAL_DIR}/command.sh' <<'CMD'
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='${GPUS}'
export PATH='${VENV}/bin':"\$PATH"
export PYTHONPATH='${REMOTE_ROOT}/external/revisiting_opd':"\${PYTHONPATH:-}"
export HF_HOME='${HF_HOME_DIR}'
export HF_HUB_CACHE='${HF_HOME_DIR}/hub'
export HF_DATASETS_CACHE='${HF_HOME_DIR}/datasets'
export TMPDIR='${LOCAL_CACHE_ROOT}/tmp'
export VLLM_CACHE_ROOT='${LOCAL_CACHE_ROOT}/vllm_cache'
export TORCHINDUCTOR_CACHE_DIR='${LOCAL_CACHE_ROOT}/torchinductor'
export TRITON_CACHE_DIR='${LOCAL_CACHE_ROOT}/triton'
export CUDA_CACHE_PATH='${LOCAL_CACHE_ROOT}/cuda_cache'
export OUTLINES_CACHE_DIR='${LOCAL_CACHE_ROOT}/outlines'
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd '${REMOTE_ROOT}'
if [[ ! -f '${MODEL_DIR}/config.json' ]]; then
  '${VENV}/bin/python' external/revisiting_opd/scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir '${ACTOR_DIR}' \
    --target_dir '${MODEL_DIR}'
fi
'${VENV}/bin/python' scripts/eval_qwen3_math_vllm.py \
  --model-path '${MODEL_DIR}' \
  --eval-jsonl-dir '${EVAL_DATA_DIR}' \
  --output-dir '${EVAL_DIR}/outputs' \
  --tasks ${TASKS} \
  --n '${N}' \
  --temperature '${TEMPERATURE}' \
  --top-p '${TOP_P}' \
  --max-tokens '${MAX_TOKENS}' \
  --gpus '${GPUS}' \
  --length-tokenizer-path '${LENGTH_TOKENIZER_PATH}'
CMD
chmod +x '${EVAL_DIR}/command.sh'
if [[ -f '${EVAL_DIR}/eval.pid' ]] && ps -p \"\$(cat '${EVAL_DIR}/eval.pid')\" >/dev/null 2>&1; then
  echo \"already_running pid=\$(cat '${EVAL_DIR}/eval.pid') eval_dir=${EVAL_DIR}\"
  exit 0
fi
nohup bash '${EVAL_DIR}/command.sh' > '${LOG_DIR}/eval.log' 2>&1 &
echo \$! > '${EVAL_DIR}/eval.pid'
echo \"pid=\$(cat '${EVAL_DIR}/eval.pid') eval_dir=${EVAL_DIR} log=${LOG_DIR}/eval.log\"
"
