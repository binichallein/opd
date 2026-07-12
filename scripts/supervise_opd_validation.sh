#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EVAL_STEPS=(50 100 200)

case "${TARGET}" in
  train-pair)
    REMOTE="train"
    RUN_ROOT="/mnt/data/cpfs/Yaleon/opd/runs/20260712v1_deepseek_justrl_pair_seed21_train"
    VARIANTS=(token_opd block3_mean)
    ;;
  ml2-token)
    REMOTE="ml2"
    RUN_ROOT="/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2"
    VARIANTS=(token_opd)
    ;;
  *)
    echo "usage: $0 {train-pair|ml2-token}" >&2
    exit 2
    ;;
esac

LOCK_FILE="/tmp/opd_validation_supervisor_${TARGET}.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "supervisor already active for ${TARGET}" >&2
  exit 1
fi

log() {
  printf '%s target=%s %s\n' "$(date --iso-8601=seconds)" "${TARGET}" "$*"
}

control_action() {
  local variant="$1"
  local action="$2"
  local step="${3:-}"
  local args=()
  if [[ -n "${step}" ]]; then
    args+=("${step}")
  fi
  if [[ "${TARGET}" == "train-pair" ]]; then
    bash "${ROOT_DIR}/scripts/deepseek_justrl_train_control.sh" \
      "${variant}" "${action}" "${args[@]}"
  else
    bash "${ROOT_DIR}/scripts/token_opd_ml2_control.sh" \
      "${action}" "${args[@]}"
  fi
}

remote_dir_exists() {
  ssh "${REMOTE}" "test -d '$1'"
}

remote_file_exists() {
  ssh "${REMOTE}" "test -f '$1'"
}

wait_for_success() {
  local run_dir="$1"
  local pid
  local pid_file
  local code
  while ! remote_file_exists "${run_dir}/exit_code.txt"; do
    if remote_file_exists "${run_dir}/train.pid"; then
      pid_file="${run_dir}/train.pid"
    elif remote_file_exists "${run_dir}/eval.pid"; then
      pid_file="${run_dir}/eval.pid"
    else
      log "cannot continue: ${run_dir} has no train.pid, eval.pid, or exit_code.txt"
      return 1
    fi
    pid="$(ssh "${REMOTE}" "cat '${pid_file}'")"
    if ! ssh "${REMOTE}" "kill -0 '${pid}' 2>/dev/null"; then
      log "cannot continue: pid ${pid} ended without exit_code.txt for ${run_dir}"
      return 1
    fi
    log "waiting pid=${pid} run=${run_dir}"
    sleep "${POLL_SECONDS}"
  done
  code="$(ssh "${REMOTE}" "cat '${run_dir}/exit_code.txt'")"
  if [[ "${code}" != "0" ]]; then
    log "cannot continue: exit_code=${code} run=${run_dir}"
    return 1
  fi
  log "completed run=${run_dir}"
}

wait_for_idle() {
  while ssh "${REMOTE}" \
    "ps -eo cmd | grep -E '[m]ain_ppo_multitask|[r]aylet|[r]ay::|[V]LLMWorker|[e]val_qwen3_math_vllm' >/dev/null"; do
    log "waiting for ${REMOTE} to become idle"
    sleep "${POLL_SECONDS}"
  done
}

ensure_formal() {
  local variant="$1"
  local run_dir="${RUN_ROOT}/${variant}"
  if ! remote_dir_exists "${run_dir}"; then
    wait_for_idle
    log "launching formal variant=${variant}"
    control_action "${variant}" formal
  fi
  wait_for_success "${run_dir}"
  control_action "${variant}" audit-checkpoints
}

ensure_eval() {
  local variant="$1"
  local step="$2"
  local eval_dir="${RUN_ROOT}/${variant}/eval_step_${step}_n8"
  if ! remote_dir_exists "${eval_dir}"; then
    wait_for_idle
    log "launching eval variant=${variant} step=${step}"
    control_action "${variant}" eval "${step}"
  fi
  wait_for_success "${eval_dir}"
}

for variant in "${VARIANTS[@]}"; do
  ensure_formal "${variant}"
done

for variant in "${VARIANTS[@]}"; do
  for step in "${EVAL_STEPS[@]}"; do
    ensure_eval "${variant}" "${step}"
  done
  control_action "${variant}" audit
done

log "all training, checkpoint audits, full evaluations, and final audits passed"
