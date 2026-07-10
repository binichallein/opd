#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-status}"
STEP="${2:-200}"
DATE_TAG="${DATE_TAG:-20260710}"
DIAG_STRIDE="${DIAG_STRIDE:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-18432}"
SOURCE_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
SUBMODULE_BASE_COMMIT="$(git -C "${ROOT_DIR}/external/revisiting_opd" rev-parse HEAD)"

run_root_for() {
  local host="$1"
  local kind="$2"
  if [[ "${host}" == "train" ]]; then
    if [[ "${kind}" == "probe" ]]; then
      echo "/mnt/data/cpfs/Yaleon/opd/runs/${DATE_TAG}_block10_collapse_diag_probe_train"
    else
      echo "/mnt/data/cpfs/Yaleon/opd/runs/${DATE_TAG}_block10_collapse_diag_seed21_train"
    fi
  else
    if [[ "${kind}" == "probe" ]]; then
      echo "/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/${DATE_TAG}_block10_collapse_diag_probe_ml2"
    else
      echo "/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/${DATE_TAG}_block10_collapse_diag_seed21_ml2"
    fi
  fi
}

launch_host() {
  local host="$1"
  local kind="$2"
  local total_steps="$3"
  local milestones="$4"
  local interval="$5"
  local stride="$6"
  local remote_root venv hf_home student teacher cache run_root

  if [[ "${host}" == "train" ]]; then
    remote_root="/mnt/data/cpfs/Yaleon/opd"
    venv="/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/venv"
    hf_home="/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/hf_home"
    student="/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-1.7B-Base"
    teacher="/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-4B-Base-GRPO"
    cache="/tmp/opd_block10_diag_${DATE_TAG}"
  else
    remote_root="/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd"
    venv="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl"
    hf_home="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/hf_home"
    student="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/models/Qwen3-1.7B-Base"
    teacher="${remote_root}/models/Qwen3-4B-Base-GRPO"
    cache="/limx_embap/tos/b10diag_${DATE_TAG}"
  fi
  run_root="$(run_root_for "${host}" "${kind}")"

  REMOTE="${host}" \
  REMOTE_ROOT="${remote_root}" \
  RUN_ROOT="${run_root}" \
  VARIANT="block10_mean" \
  PROJECT_NAME="opd_block10_collapse_diagnostics" \
  EXP_NAME="block10-mean-diag-${host}-${kind}" \
  VENV="${venv}" \
  HF_HOME_DIR="${hf_home}" \
  LOCAL_CACHE_ROOT="${cache}" \
  STUDENT_MODEL="${student}" \
  MATH_TEACHER="${teacher}" \
  ENV_SEED=21 \
  TRAIN_BATCH_SIZE=4 \
  PPO_MINI_BATCH_SIZE=32 \
  ROLLOUT_GROUP_SIZE=8 \
  MAX_PROMPT_LENGTH=2048 \
  MAX_RESPONSE_LENGTH=16384 \
  LEARNING_RATE=2e-6 \
  TOTAL_TRAINING_STEPS="${total_steps}" \
  SAVE_FREQ=-1 \
  TEST_FREQ=1000000 \
  VAL_N=1 \
  ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  OPD_DIAGNOSTICS=true \
  SOURCE_COMMIT="${SOURCE_COMMIT}" \
  SUBMODULE_BASE_COMMIT="${SUBMODULE_BASE_COMMIT}" \
  OPD_DIAG_INTERVAL="${interval}" \
  OPD_DIAG_TOPK=16 \
  OPD_DIAG_POSITION_BIN=128 \
  OPD_DIAG_POSITION_STRIDE="${stride}" \
  OPD_DIAG_SIGN_EPS=1e-4 \
  DIAGNOSTIC_SAVE_STEPS="${milestones}" \
  bash "${ROOT_DIR}/scripts/launch_revisiting_block_opd_formal_train.sh"
}

status_host() {
  local host="$1"
  local run_root
  run_root="$(run_root_for "${host}" formal)/block10_mean"
  ssh "${host}" "set -euo pipefail
    echo host='${host}' run='${run_root}'
    if [[ -f '${run_root}/train.pid' ]]; then
      pid=\$(cat '${run_root}/train.pid')
      ps -p \"\${pid}\" -o pid,etime,stat,cmd || true
    fi
    find '${run_root}/checkpoints' -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V || true
    tail -n 4 '${run_root}/logs/nohup.log' 2>/dev/null || true"
}

eval_host() {
  local host="$1"
  local remote_root venv hf_home cache student run_root
  run_root="$(run_root_for "${host}" formal)"
  if [[ "${host}" == "train" ]]; then
    remote_root="/mnt/data/cpfs/Yaleon/opd"
    venv="/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/venv"
    hf_home="/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/hf_home"
    student="/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-1.7B-Base"
    cache="/tmp/opd_block10_diag_eval_${DATE_TAG}_${STEP}"
  else
    remote_root="/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd"
    venv="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl"
    hf_home="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/hf_home"
    student="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/models/Qwen3-1.7B-Base"
    cache="/limx_embap/tos/b10diag_eval_${DATE_TAG}_${STEP}"
  fi
  REMOTE="${host}" REMOTE_ROOT="${remote_root}" RUN_ROOT="${run_root}" \
    VARIANT=block10_mean STEP="${STEP}" VENV="${venv}" HF_HOME_DIR="${hf_home}" \
    LOCAL_CACHE_ROOT="${cache}" LENGTH_TOKENIZER_PATH="${student}" \
    N=8 TEMPERATURE=1.0 TOP_P=0.9 MAX_TOKENS=16384 \
    TASKS="math500 aime24 aime25 amc23" GPUS="0,1,2,3" \
    bash "${ROOT_DIR}/scripts/launch_qwen3_math_eval.sh"
}

case "${ACTION}" in
  sync)
    bash "${ROOT_DIR}/scripts/sync_block10_diagnostics_to_hosts.sh"
    ;;
  probe1)
    launch_host train probe 1 1 1 1
    launch_host ml2 probe 1 1 1 1
    ;;
  probe2)
    launch_host train probe 2 1,2 1 1
    launch_host ml2 probe 2 1,2 1 1
    ;;
  formal)
    launch_host train formal 200 40,50,60,80,100,200 5 "${DIAG_STRIDE}"
    launch_host ml2 formal 200 40,50,60,80,100,200 5 "${DIAG_STRIDE}"
    ;;
  status)
    status_host train
    status_host ml2
    ;;
  eval)
    eval_host train
    eval_host ml2
    ;;
  *)
    echo "usage: $0 {sync|probe1|probe2|formal|status|eval [step]}" >&2
    exit 2
    ;;
esac
