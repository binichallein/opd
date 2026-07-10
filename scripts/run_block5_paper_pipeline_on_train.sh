#!/usr/bin/env bash
set -euo pipefail

# This script is intended to run on the train host.

ROOT="${ROOT:-/mnt/data/cpfs/Yaleon/opd}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/20260709_block5_dapo17k_paper_qwen3}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-${ROOT}/runs/20260708_block3_dapo17k_paper_qwen3}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/math_opd_dapo17k_hf_full_eval4}"
VENV="${VENV:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/venv}"
HF_HOME_DIR="${HF_HOME_DIR:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/hf_home}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/tmp/opd_block5_dapo17k_pipeline}"
STUDENT_MODEL="${STUDENT_MODEL:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-1.7B-Base}"
MATH_TEACHER="${MATH_TEACHER:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-4B-Base-GRPO}"

TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.parquet}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/test.parquet}"
EVAL_JSONL_DIR="${EVAL_JSONL_DIR:-${DATA_DIR}/eval_jsonl}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
ENV_SEED="${ENV_SEED:-21}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_GROUP_SIZE="${ROLLOUT_GROUP_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-1000000}"
VAL_N="${VAL_N:-1}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console']}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-18432}"

EVAL_N="${EVAL_N:-8}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-16384}"
EVAL_TASKS="${EVAL_TASKS:-math500 aime24 aime25 amc23}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"

mkdir -p "${RUN_ROOT}/pipeline_logs"
PIPELINE_LOG="${RUN_ROOT}/pipeline_logs/pipeline.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PIPELINE_LOG}"
}

pid_alive() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] && ps -p "$(cat "${pid_file}")" >/dev/null 2>&1
}

checkpoint_complete() {
  local variant="$1"
  local step="$2"
  local ckpt="${RUN_ROOT}/${variant}/checkpoints/global_step_${step}"
  local actor="${ckpt}/actor"
  [[ -d "${actor}" ]] || return 1
  [[ -f "${ckpt}/data.pt" ]] || return 1
  [[ "$(find "${actor}" -maxdepth 1 -name 'model_world_size_*_rank_*.pt' | wc -l)" -ge "${N_GPUS_PER_NODE}" ]] || return 1
  [[ "$(find "${actor}" -maxdepth 1 -name 'optim_world_size_*_rank_*.pt' | wc -l)" -ge "${N_GPUS_PER_NODE}" ]] || return 1
  [[ "$(find "${actor}" -maxdepth 1 -name 'extra_state_world_size_*_rank_*.pt' | wc -l)" -ge "${N_GPUS_PER_NODE}" ]] || return 1
}

write_train_command() {
  local variant="$1"
  local run_dir="${RUN_ROOT}/${variant}"
  local ckpts_dir="${run_dir}/checkpoints"
  local log_dir="${run_dir}/logs"
  local exp_name="${variant}-qwen3-dapo17k-paper"
  mkdir -p "${run_dir}" "${ckpts_dir}" "${log_dir}" "${run_dir}/summaries"
  cp "${DATA_DIR}/manifest.json" "${run_dir}/data_manifest.json"
  sha256sum \
    "${ROOT}/scripts/run_revisiting_sampled_block_opd_math.sh" \
    "${ROOT}/external/revisiting_opd/verl/trainer/ppo/core_algos.py" \
    "${ROOT}/external/revisiting_opd/verl/workers/actor/dp_actor.py" \
    "${ROOT}/external/revisiting_opd/verl/trainer/config/ppo_trainer.yaml" \
    > "${run_dir}/script_hashes.sha256"
  cat > "${run_dir}/run_card.json" <<JSON
{
  "variant": "${variant}",
  "project_name": "opd_block5_dapo17k",
  "experiment_name": "${exp_name}",
  "student_model": "${STUDENT_MODEL}",
  "teacher_model": "${MATH_TEACHER}",
  "venv": "${VENV}",
  "train_data": "${TRAIN_DATA}",
  "val_data": "${VAL_DATA}",
  "seed": ${ENV_SEED},
  "n_gpus_per_node": ${N_GPUS_PER_NODE},
  "ray_num_cpus": ${RAY_NUM_CPUS},
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "ppo_mini_batch_size": ${PPO_MINI_BATCH_SIZE},
  "rollout_group_size": ${ROLLOUT_GROUP_SIZE},
  "max_prompt_length": ${MAX_PROMPT_LENGTH},
  "max_response_length": ${MAX_RESPONSE_LENGTH},
  "learning_rate": ${LEARNING_RATE},
  "total_training_steps": ${TOTAL_TRAINING_STEPS},
  "save_freq": ${SAVE_FREQ},
  "test_freq": ${TEST_FREQ},
  "val_n": ${VAL_N},
  "rollout_gpu_memory_utilization": ${ROLLOUT_GPU_MEMORY_UTILIZATION},
  "rollout_max_num_batched_tokens": ${ROLLOUT_MAX_NUM_BATCHED_TOKENS},
  "checkpoint_policy": "preserve all saved checkpoints; no save_total_limit; save every 100 steps and final step",
  "baseline_run_root": "${BASELINE_RUN_ROOT}",
  "baseline_alignment": "Compare with completed token_opd and block3_mean from the paper-aligned Qwen3/DAPO run; train only block5_mean here"
}
JSON
  cat > "${run_dir}/command.sh" <<CMD
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PATH='${VENV}/bin':"\$PATH"
export PYTHONPATH='${ROOT}/external/revisiting_opd':"\${PYTHONPATH:-}"
export HF_HOME='${HF_HOME_DIR}'
export HF_HUB_CACHE='${HF_HOME_DIR}/hub'
export HF_DATASETS_CACHE='${HF_HOME_DIR}/datasets'
export TMPDIR='${LOCAL_CACHE_ROOT}/tmp'
export VLLM_CACHE_ROOT='${LOCAL_CACHE_ROOT}/vllm_cache'
export TORCHINDUCTOR_CACHE_DIR='${LOCAL_CACHE_ROOT}/torchinductor'
export TRITON_CACHE_DIR='${LOCAL_CACHE_ROOT}/triton'
export CUDA_CACHE_PATH='${LOCAL_CACHE_ROOT}/cuda_cache'
export OUTLINES_CACHE_DIR='${LOCAL_CACHE_ROOT}/outlines'
export PYTHONUNBUFFERED=1
mkdir -p '${LOCAL_CACHE_ROOT}/tmp' '${LOCAL_CACHE_ROOT}/vllm_cache' '${LOCAL_CACHE_ROOT}/torchinductor' '${LOCAL_CACHE_ROOT}/triton' '${LOCAL_CACHE_ROOT}/cuda_cache' '${LOCAL_CACHE_ROOT}/outlines'
cd '${ROOT}'
VARIANT='${variant}' \
PROJECT_NAME='opd_block5_dapo17k' \
EXP_NAME='${exp_name}' \
STUDENT_MODEL='${STUDENT_MODEL}' \
MATH_TEACHER='${MATH_TEACHER}' \
TRAIN_DATA='${TRAIN_DATA}' \
VAL_DATA='${VAL_DATA}' \
CKPTS_DIR='${ckpts_dir}' \
LOG_DIR='${log_dir}' \
N_GPUS_PER_NODE='${N_GPUS_PER_NODE}' \
RAY_NUM_CPUS='${RAY_NUM_CPUS}' \
ENV_SEED='${ENV_SEED}' \
TRAIN_BATCH_SIZE='${TRAIN_BATCH_SIZE}' \
PPO_MINI_BATCH_SIZE='${PPO_MINI_BATCH_SIZE}' \
ROLLOUT_GROUP_SIZE='${ROLLOUT_GROUP_SIZE}' \
ROLLOUT_GPU_MEMORY_UTILIZATION='${ROLLOUT_GPU_MEMORY_UTILIZATION}' \
ROLLOUT_MAX_NUM_BATCHED_TOKENS='${ROLLOUT_MAX_NUM_BATCHED_TOKENS}' \
MAX_PROMPT_LENGTH='${MAX_PROMPT_LENGTH}' \
MAX_RESPONSE_LENGTH='${MAX_RESPONSE_LENGTH}' \
LEARNING_RATE='${LEARNING_RATE}' \
TOTAL_TRAINING_STEPS='${TOTAL_TRAINING_STEPS}' \
SAVE_FREQ='${SAVE_FREQ}' \
TEST_FREQ='${TEST_FREQ}' \
VAL_N='${VAL_N}' \
TRAINER_LOGGER="${TRAINER_LOGGER}" \
bash scripts/run_revisiting_sampled_block_opd_math.sh
CMD
  chmod +x "${run_dir}/command.sh"
}

launch_train_variant() {
  local variant="$1"
  if checkpoint_complete "${variant}" "${TOTAL_TRAINING_STEPS}"; then
    log "${variant} checkpoint global_step_${TOTAL_TRAINING_STEPS} already complete"
    return
  fi
  local run_dir="${RUN_ROOT}/${variant}"
  if pid_alive "${run_dir}/train.pid"; then
    log "${variant} already running pid=$(cat "${run_dir}/train.pid")"
    return
  fi
  write_train_command "${variant}"
  nohup bash "${run_dir}/command.sh" > "${run_dir}/logs/nohup.log" 2>&1 &
  echo $! > "${run_dir}/train.pid"
  log "launched ${variant} pid=$(cat "${run_dir}/train.pid")"
}

wait_for_finished_variant() {
  local variant="$1"
  local step="${2:-${TOTAL_TRAINING_STEPS}}"
  local run_dir="${RUN_ROOT}/${variant}"
  while true; do
    if checkpoint_complete "${variant}" "${step}"; then
      if pid_alive "${run_dir}/train.pid"; then
        log "${variant} checkpoint exists, waiting for process to exit"
      else
        log "${variant} complete at global_step_${step}"
        return
      fi
    elif pid_alive "${run_dir}/train.pid"; then
      local latest
      latest="$(grep -o 'training/global_step:[0-9.]*' "${run_dir}/logs/nohup.log" 2>/dev/null | tail -n 1 || true)"
      log "${variant} running ${latest:-no_step_yet}"
    else
      log "ERROR: ${variant} is not running and global_step_${step} is incomplete"
      exit 1
    fi
    sleep 300
  done
}

cleanup_after_training_variant() {
  local variant="$1"
  log "cleanup after ${variant}: stopping leftover Ray workers if any"
  "${VENV}/bin/ray" stop --force >> "${PIPELINE_LOG}" 2>&1 || true
  sleep 10
}

cleanup_before_training() {
  log "preflight cleanup: stopping leftover Ray workers and removing stale local Ray state"
  "${VENV}/bin/ray" stop --force >> "${PIPELINE_LOG}" 2>&1 || true
  rm -rf "${LOCAL_CACHE_ROOT}/tmp/ray"
  mkdir -p "${LOCAL_CACHE_ROOT}/tmp"
  sleep 10
}

run_eval_variant() {
  local variant="$1"
  local step="${2:-${TOTAL_TRAINING_STEPS}}"
  local run_dir="${RUN_ROOT}/${variant}"
  local ckpt="${run_dir}/checkpoints/global_step_${step}"
  local actor="${ckpt}/actor"
  local model_dir="${actor}/huggingface"
  local eval_dir="${run_dir}/eval_step_${step}_n${EVAL_N}"
  local log_dir="${eval_dir}/logs"
  mkdir -p "${eval_dir}/outputs" "${log_dir}" \
    "${LOCAL_CACHE_ROOT}/eval_tmp" "${LOCAL_CACHE_ROOT}/eval_vllm_cache" \
    "${LOCAL_CACHE_ROOT}/eval_torchinductor" "${LOCAL_CACHE_ROOT}/eval_triton" \
    "${LOCAL_CACHE_ROOT}/eval_cuda_cache" "${LOCAL_CACHE_ROOT}/eval_outlines"
  if [[ -f "${eval_dir}/outputs/summary.json" ]]; then
    log "${variant} eval summary already exists: ${eval_dir}/outputs/summary.json"
    return
  fi
  cat > "${eval_dir}/eval_card.json" <<JSON
{
  "variant": "${variant}",
  "step": ${step},
  "checkpoint_dir": "${ckpt}",
  "model_dir": "${model_dir}",
  "eval_jsonl_dir": "${EVAL_JSONL_DIR}",
  "n": ${EVAL_N},
  "temperature": ${EVAL_TEMPERATURE},
  "top_p": ${EVAL_TOP_P},
  "max_tokens": ${EVAL_MAX_TOKENS},
  "tasks": "${EVAL_TASKS}",
  "gpus": "${EVAL_GPUS}"
}
JSON
  log "starting eval for ${variant}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${EVAL_GPUS}"
    export PATH="${VENV}/bin:$PATH"
    export PYTHONPATH="${ROOT}/external/revisiting_opd:${PYTHONPATH:-}"
    export HF_HOME="${HF_HOME_DIR}"
    export HF_HUB_CACHE="${HF_HOME_DIR}/hub"
    export HF_DATASETS_CACHE="${HF_HOME_DIR}/datasets"
    export TMPDIR="${LOCAL_CACHE_ROOT}/eval_tmp"
    export VLLM_CACHE_ROOT="${LOCAL_CACHE_ROOT}/eval_vllm_cache"
    export TORCHINDUCTOR_CACHE_DIR="${LOCAL_CACHE_ROOT}/eval_torchinductor"
    export TRITON_CACHE_DIR="${LOCAL_CACHE_ROOT}/eval_triton"
    export CUDA_CACHE_PATH="${LOCAL_CACHE_ROOT}/eval_cuda_cache"
    export OUTLINES_CACHE_DIR="${LOCAL_CACHE_ROOT}/eval_outlines"
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1
    cd "${ROOT}"
    if [[ ! -f "${model_dir}/config.json" ]]; then
      "${VENV}/bin/python" external/revisiting_opd/scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir "${actor}" \
        --target_dir "${model_dir}"
    fi
    "${VENV}/bin/python" scripts/eval_qwen3_math_vllm.py \
      --model-path "${model_dir}" \
      --eval-jsonl-dir "${EVAL_JSONL_DIR}" \
      --output-dir "${eval_dir}/outputs" \
      --tasks ${EVAL_TASKS} \
      --n "${EVAL_N}" \
      --temperature "${EVAL_TEMPERATURE}" \
      --top-p "${EVAL_TOP_P}" \
      --max-tokens "${EVAL_MAX_TOKENS}" \
      --gpus "${EVAL_GPUS}" \
      --length-tokenizer-path "${STUDENT_MODEL}"
  ) > "${log_dir}/eval.log" 2>&1
  log "finished eval for ${variant}: ${eval_dir}/outputs/summary.json"
}

main() {
  log "pipeline start"
  test -x "${VENV}/bin/python"
  test -f "${TRAIN_DATA}"
  test -f "${VAL_DATA}"
  test -f "${BASELINE_RUN_ROOT}/token_opd/eval_step_200_n8/outputs/summary.json"
  test -f "${BASELINE_RUN_ROOT}/block3_mean/eval_step_200_n8/outputs/summary.json"
  cleanup_before_training
  launch_train_variant block5_mean
  wait_for_finished_variant block5_mean "${TOTAL_TRAINING_STEPS}"
  cleanup_after_training_variant block5_mean
  run_eval_variant block5_mean "${TOTAL_TRAINING_STEPS}"
  log "pipeline complete"
}

main "$@"
