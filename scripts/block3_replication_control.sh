#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-status}"
STEP="${2:-200}"
DATE_TAG="${DATE_TAG:-20260711v1}"

REMOTE="ml2"
REMOTE_ROOT="/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd"
VENV="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl"
HF_HOME_DIR="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/hf_home"
STUDENT_MODEL="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/models/Qwen3-1.7B-Base"
MATH_TEACHER="${REMOTE_ROOT}/models/Qwen3-4B-Base-GRPO"
DATA_DIR="${REMOTE_ROOT}/data/math_opd_dapo17k_hf_full_eval4"
CACHE_ROOT="${REMOTE_ROOT}/cache/block3_replication_${DATE_TAG}"

VARIANT="block3_mean"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
FORMAL_OPD_DIAG_INTERVAL=5
SOURCE_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
SUBMODULE_BASE_COMMIT="$(git -C "${ROOT_DIR}/external/revisiting_opd" rev-parse HEAD)"

run_root_for() {
  local kind="$1"
  if [[ "${kind}" == "probe" ]]; then
    echo "${REMOTE_ROOT}/runs/${DATE_TAG}_block3_replication_probe_seed21_ml2"
  else
    echo "${REMOTE_ROOT}/runs/${DATE_TAG}_block3_replication_seed21_ml2"
  fi
}

launch_train() {
  local kind="$1"
  local total_steps="$2"
  local milestones="$3"
  local stop_after_step="$4"
  local diag_interval="${5:-${FORMAL_OPD_DIAG_INTERVAL}}"
  local run_root
  run_root="$(run_root_for "${kind}")"

  REMOTE="${REMOTE}" \
  REMOTE_ROOT="${REMOTE_ROOT}" \
  RUN_ROOT="${run_root}" \
  VARIANT="block3_mean" \
  PROJECT_NAME="opd_block3_replication" \
  EXP_NAME="block3-mean-replication-ml2-${kind}" \
  VENV="${VENV}" \
  HF_HOME_DIR="${HF_HOME_DIR}" \
  LOCAL_CACHE_ROOT="${CACHE_ROOT}/${kind}" \
  STUDENT_MODEL="${STUDENT_MODEL}" \
  MATH_TEACHER="${MATH_TEACHER}" \
  DATA_DIR="${DATA_DIR}" \
  ENV_SEED=21 \
  TRAIN_BATCH_SIZE=4 \
  PPO_MINI_BATCH_SIZE=32 \
  ROLLOUT_GROUP_SIZE=8 \
  MAX_PROMPT_LENGTH=2048 \
  MAX_RESPONSE_LENGTH=16384 \
  LEARNING_RATE=2e-6 \
  TOTAL_TRAINING_STEPS="${total_steps}" \
  SAVE_FREQ=-1 \
  TEST_FREQ=-1 \
  VAL_N=1 \
  ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS=18432 \
  ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  OPD_DIAGNOSTICS=true \
  OPD_DIAG_INTERVAL="${diag_interval}" \
  OPD_DIAG_TOPK=16 \
  OPD_DIAG_POSITION_BIN=128 \
  OPD_DIAG_POSITION_STRIDE=1 \
  OPD_DIAG_SIGN_EPS=1e-4 \
  DIAGNOSTIC_SAVE_STEPS="${milestones}" \
  STOP_AFTER_STEP="${stop_after_step}" \
  FILTER_OVERLONG_PROMPTS=false \
  SOURCE_COMMIT="${SOURCE_COMMIT}" \
  SUBMODULE_BASE_COMMIT="${SUBMODULE_BASE_COMMIT}" \
  bash "${ROOT_DIR}/scripts/launch_revisiting_block_opd_formal_train.sh"
}

status_run() {
  local run_dir
  run_dir="$(run_root_for formal)/${VARIANT}"
  ssh "${REMOTE}" "set -euo pipefail
    echo run='${run_dir}'
    if [[ -f '${run_dir}/train.pid' ]]; then
      pid=\$(cat '${run_dir}/train.pid')
      ps -p \"\${pid}\" -o pid,etime,stat,cmd || true
    fi
    printf 'exit_code='; cat '${run_dir}/exit_code.txt' 2>/dev/null || echo running
    find '${run_dir}/checkpoints' -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V || true
    find '${run_dir}/diagnostics' -maxdepth 1 -type f -name 'step_*.npz' -printf '%f\n' 2>/dev/null | sort -V | tail -n 5 || true
    tail -n 4 '${run_dir}/logs/nohup.log' 2>/dev/null || true"
}

launch_eval() {
  local run_root
  run_root="$(run_root_for formal)"
  REMOTE="${REMOTE}" \
  REMOTE_ROOT="${REMOTE_ROOT}" \
  RUN_ROOT="${run_root}" \
  VARIANT="${VARIANT}" \
  STEP="${STEP}" \
  VENV="${VENV}" \
  HF_HOME_DIR="${HF_HOME_DIR}" \
  LOCAL_CACHE_ROOT="${CACHE_ROOT}/eval_step_${STEP}" \
  EVAL_DATA_DIR="${DATA_DIR}/eval_jsonl" \
  LENGTH_TOKENIZER_PATH="${STUDENT_MODEL}" \
  N=8 TEMPERATURE=1.0 TOP_P=0.9 MAX_TOKENS=16384 \
  EVAL_SEED=21 GRADER=verl ENABLE_THINKING=false \
  TASKS="math500 aime24 aime25 amc23" GPUS="0,1,2,3" \
  bash "${ROOT_DIR}/scripts/launch_qwen3_math_eval.sh"
}

audit_formal() {
  local run_dir
  run_dir="$(run_root_for formal)/${VARIANT}"
  ssh "${REMOTE}" "'${VENV}/bin/python' '${REMOTE_ROOT}/scripts/audit_block10_run.py' \
    --run-dir '${run_dir}' --variant block3_mean --checkpoint-steps 50,100,200 --eval-steps 50,100,200"
}

case "${ACTION}" in
  sync)
    ML2_ROOT="${REMOTE_ROOT}" bash "${ROOT_DIR}/scripts/sync_block3_replication_to_ml2.sh"
    ;;
  probe1)
    launch_train probe 2 1 1 1
    ;;
  probe2)
    launch_train probe 2 1,2 -1 1
    ;;
  formal)
    launch_train formal 200 50,100,200 -1 "${FORMAL_OPD_DIAG_INTERVAL}"
    ;;
  status)
    status_run
    ;;
  eval)
    launch_eval
    ;;
  audit)
    audit_formal
    ;;
  *)
    echo "usage: $0 {sync|probe1|probe2|formal|status|eval [step]|audit}" >&2
    exit 2
    ;;
esac
