#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SSH_BIN="${SSH_BIN:-ssh}"
CONTROL_DRIVER="${CONTROL_DRIVER:-}"
EVAL_STEPS=(50 100 200)

case "${TARGET}" in
  train-pair)
    REMOTE="${REMOTE_OVERRIDE:-train}"
    RUN_ROOT="${RUN_ROOT_OVERRIDE:-/mnt/data/cpfs/Yaleon/opd/runs/20260712v1_deepseek_justrl_pair_seed21_train}"
    VARIANTS=(token_opd block3_mean)
    ;;
  ml2-token)
    REMOTE="${REMOTE_OVERRIDE:-ml2}"
    RUN_ROOT="${RUN_ROOT_OVERRIDE:-/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2}"
    VARIANTS=(token_opd)
    ;;
  *)
    echo "usage: $0 {train-pair|ml2-token}" >&2
    exit 2
    ;;
esac

LOCK_FILE="${LOCK_FILE_OVERRIDE:-/tmp/opd_validation_supervisor_${TARGET}.lock}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "supervisor already active for ${TARGET}" >&2
  exit 1
fi

log() {
  printf '%s target=%s %s\n' "$(date --iso-8601=seconds)" "${TARGET}" "$*"
}

REMOTE_LOCK_HELD=0

ensure_remote_run_root() {
  local status
  if "${SSH_BIN}" "${REMOTE}" "mkdir -p -- '${RUN_ROOT}'"; then
    return 0
  else
    status=$?
  fi
  log "SSH failure status=${status} while creating run root ${RUN_ROOT}"
  return 2
}

control_action() {
  local variant="$1"
  local action="$2"
  local step="${3:-}"
  local args=()
  if [[ -n "${step}" ]]; then
    args+=("${step}")
  fi
  if [[ -n "${CONTROL_DRIVER}" ]]; then
    "${CONTROL_DRIVER}" "${TARGET}" "${variant}" "${action}" "${args[@]}"
  elif [[ "${TARGET}" == "train-pair" ]]; then
    bash "${ROOT_DIR}/scripts/deepseek_justrl_train_control.sh" \
      "${variant}" "${action}" "${args[@]}"
  else
    bash "${ROOT_DIR}/scripts/token_opd_ml2_control.sh" \
      "${action}" "${args[@]}"
  fi
}

remote_dir_state() {
  local path="$1"
  local status
  if "${SSH_BIN}" "${REMOTE}" "test -d '${path}'"; then
    return 0
  else
    status=$?
  fi
  if [[ "${status}" == "1" ]]; then
    return 1
  fi
  log "SSH failure status=${status} while checking directory ${path}"
  return 2
}

remote_run_state() {
  local run_dir="$1"
  local output
  local status
  if output="$("${SSH_BIN}" "${REMOTE}" bash -s -- "${run_dir}" <<'REMOTE_STATE'
set -u
run_dir="$1"
expected_command="${run_dir}/command.sh"
pid_file=""
if [[ -f "${run_dir}/train.pid" ]]; then
  pid_file="${run_dir}/train.pid"
elif [[ -f "${run_dir}/eval.pid" ]]; then
  pid_file="${run_dir}/eval.pid"
fi
pid=""
active=false
identity=false
if [[ -n "${pid_file}" ]]; then
  pid="$(cat "${pid_file}")"
  process_state="$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
  if [[ "${process_state}" != "Z" ]] && kill -0 "${pid}" 2>/dev/null; then
    cmdline="$(cat "/proc/${pid}/cmdline" 2>/dev/null | tr '\0' ' ' || true)"
    if [[ -n "${cmdline}" ]]; then
      active=true
      if [[ "${cmdline}" == *"${expected_command}"* ]]; then
        identity=true
      fi
    fi
  fi
fi
if [[ "${active}" == true ]]; then
  if [[ "${identity}" == true ]]; then
    echo running
  else
    echo pid_mismatch
  fi
elif [[ -f "${run_dir}/exit_code.txt" ]]; then
  code="$(cat "${run_dir}/exit_code.txt")"
  if [[ "${code}" == "0" ]]; then
    echo success
  else
    echo "failed:${code}"
  fi
elif [[ -z "${pid_file}" ]]; then
  echo missing
elif [[ "${active}" != true ]]; then
  echo orphan
fi
REMOTE_STATE
)"; then
    printf '%s\n' "${output}"
    return 0
  else
    status=$?
  fi
  log "SSH failure status=${status} while reading run state ${run_dir}"
  return 2
}

wait_for_success() {
  local run_dir="$1"
  local state
  local status
  local transient_failures=0
  while true; do
    if state="$(remote_run_state "${run_dir}")"; then
      :
    else
      status=$?
      return "${status}"
    fi
    case "${state}" in
      running)
        transient_failures=0
        log "waiting run=${run_dir}"
        sleep "${POLL_SECONDS}"
        ;;
      success)
        log "completed run=${run_dir}"
        return 0
        ;;
      missing|orphan)
        transient_failures=$((transient_failures + 1))
        if ((transient_failures >= 3)); then
          log "cannot continue: state=${state} run=${run_dir}"
          return 1
        fi
        sleep 1
        ;;
      failed:*|pid_mismatch)
        log "cannot continue: state=${state} run=${run_dir}"
        return 1
        ;;
      *)
        log "cannot continue: unknown state=${state} run=${run_dir}"
        return 1
        ;;
    esac
  done
}

wait_for_idle() {
  local status
  while true; do
    if "${SSH_BIN}" "${REMOTE}" bash -s <<'REMOTE_IDLE'
ps -eo cmd | grep -E '[m]ain_ppo_multitask|[r]aylet|[r]ay::|[V]LLMWorker|[e]val_qwen3_math_vllm' >/dev/null
REMOTE_IDLE
    then
      log "waiting for ${REMOTE} to become idle"
      sleep "${POLL_SECONDS}"
    else
      status=$?
      if [[ "${status}" == "1" ]]; then
        return 0
      fi
      log "SSH failure status=${status} while checking whether ${REMOTE} is idle"
      return 2
    fi
  done
}

acquire_launch_lock() {
  local lock_dir="${RUN_ROOT}/.supervisor-launch.lock"
  local state
  local status
  if "${SSH_BIN}" "${REMOTE}" "mkdir '${lock_dir}'"; then
    REMOTE_LOCK_HELD=1
    log "acquired remote launch lock ${lock_dir}"
    return 0
  else
    status=$?
  fi
  if remote_dir_state "${lock_dir}"; then
    log "cannot continue: remote launch lock is already held: ${lock_dir}"
    return 1
  else
    state=$?
  fi
  if [[ "${state}" == "2" ]]; then
    return 2
  fi
  log "cannot continue: mkdir failed status=${status} for ${lock_dir}"
  return 1
}

release_launch_lock() {
  local lock_dir="${RUN_ROOT}/.supervisor-launch.lock"
  local status
  if [[ "${REMOTE_LOCK_HELD}" != "1" ]]; then
    return 0
  fi
  if "${SSH_BIN}" "${REMOTE}" "rmdir '${lock_dir}'"; then
    REMOTE_LOCK_HELD=0
    log "released remote launch lock ${lock_dir}"
    return 0
  else
    status=$?
  fi
  log "SSH failure or stale lock status=${status} while releasing ${lock_dir}"
  return 2
}

cleanup_remote_launch_lock() {
  local original_status=$?
  if [[ "${REMOTE_LOCK_HELD}" == "1" ]]; then
    release_launch_lock || true
  fi
  return "${original_status}"
}

handle_signal() {
  local exit_status="$1"
  trap - INT TERM
  exit "${exit_status}"
}

trap cleanup_remote_launch_lock EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

launch_if_missing() {
  local target_dir="$1"
  local variant="$2"
  local action="$3"
  local step="${4:-}"
  local state
  local result=0

  wait_for_idle
  acquire_launch_lock
  if wait_for_idle; then
    :
  else
    result=$?
  fi
  if [[ "${result}" == "0" ]]; then
    if remote_dir_state "${target_dir}"; then
      log "target appeared while acquiring lock; skipping launch ${target_dir}"
    else
      state=$?
      if [[ "${state}" == "1" ]]; then
        log "launching action=${action} variant=${variant} step=${step:-none}"
        if control_action "${variant}" "${action}" "${step}"; then
          if remote_dir_state "${target_dir}"; then
            :
          else
            state=$?
            log "cannot continue: launch returned but target state=${state}: ${target_dir}"
            result=1
          fi
        else
          result=$?
        fi
      else
        result="${state}"
      fi
    fi
  fi
  if ! release_launch_lock; then
    return 2
  fi
  return "${result}"
}

final_acceptance_state() {
  local run_dir="$1"
  local status
  if "${SSH_BIN}" "${REMOTE}" python3 - "${run_dir}/acceptance.json" <<'REMOTE_ACCEPTANCE'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    data = json.loads(path.read_text())
except Exception as error:
    print(f"invalid acceptance JSON: {error}", file=sys.stderr)
    raise SystemExit(2)
required = {50, 100, 200}
passed = data.get("passed") is True and not data.get("issues")
complete = required.issubset(set(data.get("checkpoint_steps", []))) and required.issubset(
    set(data.get("eval_steps", []))
)
raise SystemExit(0 if passed and complete else 1)
REMOTE_ACCEPTANCE
  then
    return 0
  else
    status=$?
  fi
  if [[ "${status}" == "1" ]]; then
    return 1
  fi
  log "SSH failure or invalid acceptance status=${status} run=${run_dir}"
  return 2
}

preserve_checkpoint_acceptance() {
  local run_dir="$1"
  local status
  if "${SSH_BIN}" "${REMOTE}" bash -s -- "${run_dir}" <<'REMOTE_PRESERVE'
set -euo pipefail
run_dir="$1"
source="${run_dir}/acceptance.json"
target="${run_dir}/checkpoint_acceptance.json"
test -f "${source}"
if [[ -f "${target}" ]]; then
  cmp -s "${source}" "${target}"
else
  cp -- "${source}" "${target}"
fi
REMOTE_PRESERVE
  then
    return 0
  else
    status=$?
  fi
  log "SSH failure or checkpoint acceptance mismatch status=${status} run=${run_dir}"
  return 2
}

ensure_formal() {
  local variant="$1"
  local run_dir="${RUN_ROOT}/${variant}"
  local state
  local acceptance_state
  if remote_dir_state "${run_dir}"; then
    :
  else
    state=$?
    if [[ "${state}" == "1" ]]; then
      launch_if_missing "${run_dir}" "${variant}" formal
    else
      return "${state}"
    fi
  fi
  wait_for_success "${run_dir}"
  if final_acceptance_state "${run_dir}"; then
    log "run already has complete final acceptance; skipping checkpoint-only audit ${run_dir}"
    return 0
  else
    acceptance_state=$?
  fi
  if [[ "${acceptance_state}" != "1" ]]; then
    return "${acceptance_state}"
  fi
  control_action "${variant}" audit-checkpoints
  preserve_checkpoint_acceptance "${run_dir}"
}

ensure_eval() {
  local variant="$1"
  local step="$2"
  local run_dir="${RUN_ROOT}/${variant}"
  local eval_dir="${run_dir}/eval_step_${step}_n8"
  local state
  local acceptance_state
  if remote_dir_state "${eval_dir}"; then
    :
  else
    state=$?
    if [[ "${state}" != "1" ]]; then
      return "${state}"
    fi
    if final_acceptance_state "${run_dir}"; then
      log "cannot continue: final acceptance exists but eval directory is missing ${eval_dir}"
      return 1
    else
      acceptance_state=$?
    fi
    if [[ "${acceptance_state}" != "1" ]]; then
      return "${acceptance_state}"
    fi
    launch_if_missing "${eval_dir}" "${variant}" eval "${step}"
  fi
  wait_for_success "${eval_dir}"
}

ensure_final_audit() {
  local variant="$1"
  local run_dir="${RUN_ROOT}/${variant}"
  local state
  if final_acceptance_state "${run_dir}"; then
    log "run already has complete final acceptance; skipping final reaudit ${run_dir}"
    return 0
  else
    state=$?
  fi
  if [[ "${state}" != "1" ]]; then
    return "${state}"
  fi
  control_action "${variant}" audit
  if final_acceptance_state "${run_dir}"; then
    return 0
  else
    state=$?
    log "cannot continue: final audit did not produce complete acceptance ${run_dir}"
    return "${state}"
  fi
}

ensure_remote_run_root

for variant in "${VARIANTS[@]}"; do
  ensure_formal "${variant}"
done

for variant in "${VARIANTS[@]}"; do
  for step in "${EVAL_STEPS[@]}"; do
    ensure_eval "${variant}" "${step}"
  done
  ensure_final_audit "${variant}"
done

log "all training, checkpoint audits, full evaluations, and final audits passed"
