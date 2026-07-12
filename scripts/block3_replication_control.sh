#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-status}"
STEP="${2:-200}"
DATE_TAG="${DATE_TAG:-20260711v2}"

REMOTE="ml2"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"
ASSET_ROOT="/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd"
RUNTIME_ROOT="${ASSET_ROOT}/deployments/${SOURCE_COMMIT}"
REMOTE_ROOT="${RUNTIME_ROOT}"
VENV="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl"
HF_HOME_DIR="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/hf_home"
STUDENT_MODEL="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/models/Qwen3-1.7B-Base"
MATH_TEACHER="${ASSET_ROOT}/models/Qwen3-4B-Base-GRPO"
DATA_DIR="${ASSET_ROOT}/data/math_opd_dapo17k_hf_full_eval4"
CACHE_ROOT="${CACHE_ROOT:-/limx_embap/tos/b3r/${DATE_TAG#2026}}"
EXPECTED_TRAIN_SHA256="cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf"
MIN_TOS_AVAILABLE_BYTES=100000000000

VARIANT="${VARIANT:-block3_mean}"
RUN_TAG="${RUN_TAG:-block3_replication}"
PROJECT_NAME="${PROJECT_NAME:-opd_block3_replication}"
EXP_PREFIX="${EXP_PREFIX:-block3-mean-replication-ml2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
FORMAL_OPD_DIAG_INTERVAL=5
SUBMODULE_BASE_COMMIT="$(git -C "${ROOT_DIR}/external/revisiting_opd" rev-parse HEAD)"

run_root_for() {
  local kind="$1"
  if [[ "${kind}" == "probe" ]]; then
    echo "${ASSET_ROOT}/runs/${DATE_TAG}_${RUN_TAG}_probe_seed21_ml2"
  else
    echo "${ASSET_ROOT}/runs/${DATE_TAG}_${RUN_TAG}_seed21_ml2"
  fi
}

run_dir_for() {
  local kind="$1"
  echo "$(run_root_for "${kind}")/${VARIANT}"
}

launch_train() {
  local kind="$1"
  local total_steps="$2"
  local milestones="$3"
  local stop_after_step="$4"
  local diag_interval="${5:-${FORMAL_OPD_DIAG_INTERVAL}}"
  local resume_mode="${6:-disable}"
  local resume_from_path="${7:-}"
  local run_root
  run_root="$(run_root_for "${kind}")"

  REMOTE="${REMOTE}" \
  REMOTE_ROOT="${REMOTE_ROOT}" \
  RUN_ROOT="${run_root}" \
  VARIANT="${VARIANT}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  EXP_NAME="${EXP_PREFIX}-${kind}" \
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
  ROLLOUT_TEMPERATURE=1.0 \
  ROLLOUT_TOP_P=0.9 \
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
  EXPECTED_TRAIN_SHA256="${EXPECTED_TRAIN_SHA256}" \
  RESUME_MODE="${resume_mode}" \
  RESUME_FROM_PATH="${resume_from_path}" \
  SOURCE_COMMIT="${SOURCE_COMMIT}" \
  SUBMODULE_BASE_COMMIT="${SUBMODULE_BASE_COMMIT}" \
  bash "${ROOT_DIR}/scripts/launch_revisiting_block_opd_formal_train.sh"
}

assert_checkpoint_complete() {
  local kind="$1"
  local step="$2"
  local run_dir
  run_dir="$(run_dir_for "${kind}")"
  ssh "${REMOTE}" "set -euo pipefail
    checkpoint='${run_dir}/checkpoints/global_step_${step}'
    test -s \"\${checkpoint}/data.pt\"
    for prefix in model optim extra_state; do
      for rank in 0 1 2 3; do
        test -s \"\${checkpoint}/actor/\${prefix}_world_size_4_rank_\${rank}.pt\"
      done
    done"
}

assert_run_stopped_successfully() {
  local kind="$1"
  local run_dir
  run_dir="$(run_dir_for "${kind}")"
  ssh "${REMOTE}" "set -euo pipefail
    test -f '${run_dir}/train.pid'
    pid=\$(cat '${run_dir}/train.pid')
    if ps -p \"\${pid}\" >/dev/null 2>&1; then
      echo 'run is still active: pid='\"\${pid}\" >&2
      exit 1
    fi
    test \"\$(cat '${run_dir}/exit_code.txt')\" = 0"
}

machine_preflight() {
  ssh "${REMOTE}" "set -euo pipefail
    test \"\$(cat '${RUNTIME_ROOT}/DEPLOYED_COMMIT')\" = '${SOURCE_COMMIT}'
    test -w '${ASSET_ROOT}'
    available=\$(df -PB1 '${ASSET_ROOT}' | awk 'NR == 2 {print \$4}')
    test \"\${available}\" -ge '${MIN_TOS_AVAILABLE_BYTES}'
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | \
      awk -F, '{gsub(/ /, \"\", \$1); gsub(/ /, \"\", \$2); if (\$1 > 1024 || \$2 > 0) busy=1} END {exit busy}'
    if ps -eo cmd | grep -E '[m]ain_ppo_multitask|[r]aylet|[r]ay::|[V]LLMWorker|[e]val_qwen3_math_vllm' >/dev/null; then
      echo 'active training, Ray, or evaluation process detected' >&2
      exit 1
    fi"
}

probe1_preflight() {
  local run_dir
  run_dir="$(run_dir_for probe)"
  machine_preflight
  ssh "${REMOTE}" "test ! -e '${run_dir}'"
}

verify_probe1() {
  local run_dir
  run_dir="$(run_dir_for probe)"
  assert_run_stopped_successfully probe
  assert_checkpoint_complete probe 1
  ssh "${REMOTE}" "'${VENV}/bin/python' '${RUNTIME_ROOT}/scripts/audit_block10_run.py' \
    --run-dir '${run_dir}' \
    --variant "${VARIANT}" \
    --checkpoint-steps 1 \
    --skip-eval \
    --expected-total-training-steps 2 \
    --expected-diagnostic-steps 1 \
    --expected-diag-interval 1 \
    --expected-resume-mode disable \
    --expected-resume-from-path '' \
    --expected-source-commit '${SOURCE_COMMIT}' \
    --expected-train-sha256 '${EXPECTED_TRAIN_SHA256}' \
    --expected-eval-data-dir '${DATA_DIR}/eval_jsonl' \
    --expected-student-model-suffix Qwen3-1.7B-Base \
    --expected-teacher-model-suffix Qwen3-4B-Base-GRPO"
}

probe2_preflight() {
  local run_dir
  run_dir="$(run_dir_for probe)"
  verify_probe1
  machine_preflight
  ssh "${REMOTE}" "test ! -e '${run_dir}/checkpoints/global_step_2'"
}

verify_probe_resume() {
  local run_dir step1
  run_dir="$(run_dir_for probe)"
  step1="${run_dir}/checkpoints/global_step_1"
  assert_run_stopped_successfully probe
  assert_checkpoint_complete probe 1
  assert_checkpoint_complete probe 2
  ssh "${REMOTE}" "set -euo pipefail
    grep -F 'Resuming from ${step1}' '${run_dir}/logs/nohup.log' >/dev/null
    '${VENV}/bin/python' '${REMOTE_ROOT}/scripts/audit_block10_run.py' \
      --run-dir '${run_dir}' \
      --variant "${VARIANT}" \
      --checkpoint-steps 1,2 \
      --skip-eval \
      --expected-total-training-steps 2 \
      --expected-diagnostic-steps 1,2 \
      --expected-diag-interval 1 \
      --expected-resume-mode resume_path \
      --expected-resume-from-path '${step1}' \
      --expected-source-commit '${SOURCE_COMMIT}' \
      --expected-train-sha256 '${EXPECTED_TRAIN_SHA256}' \
      --expected-student-model-suffix Qwen3-1.7B-Base \
      --expected-teacher-model-suffix Qwen3-4B-Base-GRPO"
}

formal_preflight() {
  local run_dir
  run_dir="$(run_dir_for formal)"
  verify_probe_resume
  machine_preflight
  ssh "${REMOTE}" "set -euo pipefail
    test ! -e '${run_dir}'
    test -w '${ASSET_ROOT}'"
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
  audit_checkpoints
  machine_preflight
  assert_checkpoint_complete formal "${STEP}"
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

audit_checkpoints() {
  local run_dir
  run_dir="$(run_root_for formal)/${VARIANT}"
  ssh "${REMOTE}" "'${VENV}/bin/python' '${REMOTE_ROOT}/scripts/audit_block10_run.py' \
    --run-dir '${run_dir}' \
    --variant "${VARIANT}" \
    --checkpoint-steps 50,100,200 \
    --skip-eval \
    --expected-source-commit '${SOURCE_COMMIT}' \
    --expected-train-sha256 '${EXPECTED_TRAIN_SHA256}' \
    --expected-eval-data-dir '${DATA_DIR}/eval_jsonl' \
    --expected-student-model-suffix Qwen3-1.7B-Base \
    --expected-teacher-model-suffix Qwen3-4B-Base-GRPO"
}

audit_formal() {
  local run_dir
  run_dir="$(run_root_for formal)/${VARIANT}"
  ssh "${REMOTE}" "'${VENV}/bin/python' '${REMOTE_ROOT}/scripts/audit_block10_run.py' \
    --run-dir '${run_dir}' --variant "${VARIANT}" --checkpoint-steps 50,100,200 --eval-steps 50,100,200 \
    --expected-source-commit '${SOURCE_COMMIT}' \
    --expected-train-sha256 '${EXPECTED_TRAIN_SHA256}' \
    --expected-eval-data-dir '${DATA_DIR}/eval_jsonl' \
    --expected-student-model-suffix Qwen3-1.7B-Base \
    --expected-teacher-model-suffix Qwen3-4B-Base-GRPO"
}

case "${ACTION}" in
  sync)
    ML2_ROOT="${ASSET_ROOT}" bash "${ROOT_DIR}/scripts/sync_block3_replication_to_ml2.sh"
    ;;
  probe1)
    probe1_preflight
    launch_train probe 2 1 1 1 disable
    ;;
  probe2)
    probe2_preflight
    probe_step1="$(run_dir_for probe)/checkpoints/global_step_1"
    launch_train probe 2 1,2 -1 1 resume_path "${probe_step1}"
    ;;
  probe-audit)
    verify_probe_resume
    ;;
  formal)
    formal_preflight
    launch_train formal 200 50,100,200 -1 "${FORMAL_OPD_DIAG_INTERVAL}" disable
    ;;
  status)
    status_run
    ;;
  eval)
    launch_eval
    ;;
  audit-checkpoints)
    audit_checkpoints
    ;;
  audit)
    audit_formal
    ;;
  *)
    echo "usage: $0 {sync|probe1|probe2|probe-audit|formal|status|eval [step]|audit-checkpoints|audit}" >&2
    exit 2
    ;;
esac
