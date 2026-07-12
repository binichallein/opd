#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-}"
SSH_BIN="${SSH_BIN:-ssh}"
SCP_BIN="${SCP_BIN:-scp}"
POLL_SECONDS="${POLL_SECONDS:-300}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-172800}"
LOCAL_PLOT_PYTHON="${LOCAL_PLOT_PYTHON:-/home/tyf/miniconda3/envs/evalscope/bin/python}"

case "${TARGET}" in
  train-pair)
    REMOTE="${REMOTE_OVERRIDE:-train}"
    REMOTE_ASSET_ROOT="${REMOTE_ASSET_ROOT_OVERRIDE:-/mnt/data/cpfs/Yaleon/opd}"
    REMOTE_PYTHON="/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/venv/bin/python"
    RUN_ROOT="/mnt/data/cpfs/Yaleon/opd/runs/20260712v1_deepseek_justrl_pair_seed21_train"
    LEFT_RUN="${RUN_ROOT}/token_opd"
    RIGHT_RUN="${RUN_ROOT}/block3_mean"
    OUTPUT_ROOT="${RUN_ROOT}/analysis"
    EXTERNAL_VIEW="historical_external_grader"
    GRADER_SOURCE="/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/eval_justrl_steps_20260606/scripts/utils.py"
    REPORT_TITLE="DeepSeek / JustRL Block3 跨师生复现"
    ;;
  ml2-pair)
    REMOTE="${REMOTE_OVERRIDE:-ml2}"
    REMOTE_ASSET_ROOT="${REMOTE_ASSET_ROOT_OVERRIDE:-/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd}"
    REMOTE_PYTHON="/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl/bin/python"
    RUNS_ROOT="${REMOTE_ASSET_ROOT}/runs"
    LEFT_RUN="${RUNS_ROOT}/20260712v1_token_opd_replication_seed21_ml2/token_opd"
    RIGHT_RUN="${RUNS_ROOT}/20260711v2_block3_replication_seed21_ml2/block3_mean"
    OUTPUT_ROOT="${RUNS_ROOT}/20260712v1_token_opd_replication_seed21_ml2/paired_analysis"
    EXTERNAL_VIEW="historical_external_grader_audited"
    GRADER_SOURCE="${RIGHT_RUN}/grading/historical_utils_sha04f7.py"
    REPORT_TITLE="Qwen3 Block3 同机严格配对验证"
    ;;
  *)
    echo "usage: $0 {train-pair|ml2-pair}" >&2
    exit 2
    ;;
esac

REPORT_TITLE_B64="$(printf '%s' "${REPORT_TITLE}" | base64 | tr -d '\n')"
ANALYSIS_COMMIT="${ANALYSIS_COMMIT:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"
ANALYSIS_RELEASE="${REMOTE_ASSET_ROOT}/analysis_deployments/${ANALYSIS_COMMIT}"
ANALYSIS_STAGE="${REMOTE_ASSET_ROOT}/analysis_deployments/.stage_${ANALYSIS_COMMIT}_$$_${RANDOM}"
OWNER_TOKEN="$(hostname)-$$-${RANDOM}"
FINALIZER_LOCK_PATH="${REMOTE_ASSET_ROOT}/analysis_deployments/.analysis-control.lock"
CONTROL_LOCK_HELD=0
LOCAL_LOCK_FILE="/tmp/opd_analysis_runtime_${TARGET}.lock"
exec 8>"${LOCAL_LOCK_FILE}"
if ! flock -n 8; then
  echo "analysis finalizer already active for ${TARGET}" >&2
  exit 1
fi
FILES=(
  scripts/analysis_runtime_control.sh
  scripts/check_final_acceptance.py
  scripts/audit_block10_run.py
  scripts/regrade_opd_eval_external.py
  scripts/compare_paired_opd_evals.py
  scripts/build_paired_validation_report.py
  scripts/analyze_single_opd_diagnostics.py
  scripts/analyze_block10_collapse_diagnostics.py
  opd_ext/__init__.py
  opd_ext/analysis.py
  opd_ext/diagnostics.py
)

log() {
  printf '%s target=%s %s\n' "$(date --iso-8601=seconds)" "${TARGET}" "$*"
}

remote_control_lock_state() {
  local status
  if "${SSH_BIN}" "${REMOTE}" bash -s -- "${FINALIZER_LOCK_PATH}" "${OWNER_TOKEN}" <<'REMOTE_LOCK_STATE'
set -u
path="$1"
token="$2"
if [[ ! -e "${path}" ]]; then exit 1; fi
if [[ ! -f "${path}" ]]; then exit 3; fi
owner="$(cat "${path}" 2>/dev/null || true)"
if [[ "${owner}" == "${token}" ]]; then exit 0; fi
exit 3
REMOTE_LOCK_STATE
  then
    return 0
  else
    status=$?
  fi
  return "${status}"
}

acquire_control_lock() {
  local output
  local status
  local owner_state
  "${SSH_BIN}" "${REMOTE}" "mkdir -p '$(dirname "${FINALIZER_LOCK_PATH}")'"
  CONTROL_LOCK_HELD=1
  if output="$("${SSH_BIN}" "${REMOTE}" bash -s -- \
    "${FINALIZER_LOCK_PATH}" "${OWNER_TOKEN}" <<'REMOTE_LOCK_ACQUIRE'
set -u
path="$1"
token="$2"
if (set -o noclobber; printf '%s\n' "${token}" > "${path}") 2>/dev/null; then
  echo acquired
  exit 0
fi
owner="$(cat "${path}" 2>/dev/null || true)"
if [[ "${owner}" == "${token}" ]]; then
  echo recovered
  exit 0
fi
exit 3
REMOTE_LOCK_ACQUIRE
)"; then
    log "${output} remote analysis control lock ${FINALIZER_LOCK_PATH}"
    return 0
  else
    status=$?
  fi
  if [[ "${status}" == "3" ]]; then
    CONTROL_LOCK_HELD=0
    log "remote analysis control lock is held by another owner"
    return 1
  fi
  if remote_control_lock_state; then
    log "recovered remote analysis control lock after SSH failure"
    return 0
  else
    owner_state=$?
  fi
  if [[ "${owner_state}" == "1" || "${owner_state}" == "3" ]]; then
    CONTROL_LOCK_HELD=0
  fi
  log "failed to determine analysis control lock ownership status=${status} owner_state=${owner_state}"
  return 2
}

remove_owned_control_lock() {
  "${SSH_BIN}" "${REMOTE}" bash -s -- "${FINALIZER_LOCK_PATH}" "${OWNER_TOKEN}" <<'REMOTE_LOCK_REMOVE'
set -u
path="$1"
token="$2"
if [[ ! -e "${path}" ]]; then exit 0; fi
if [[ ! -f "${path}" ]]; then exit 3; fi
owner="$(cat "${path}" 2>/dev/null || true)"
if [[ "${owner}" != "${token}" ]]; then exit 3; fi
rm -- "${path}"
REMOTE_LOCK_REMOVE
}

release_control_lock() {
  local attempt
  local status
  local owner_state
  if [[ "${CONTROL_LOCK_HELD}" != "1" ]]; then return 0; fi
  for attempt in 1 2; do
    if remove_owned_control_lock; then
      CONTROL_LOCK_HELD=0
      log "released remote analysis control lock"
      return 0
    else
      status=$?
    fi
    if [[ "${status}" == "3" ]]; then
      CONTROL_LOCK_HELD=0
      log "refusing to remove foreign analysis control lock"
      return 2
    fi
    if remote_control_lock_state; then
      owner_state=0
    else
      owner_state=$?
    fi
    if [[ "${owner_state}" == "1" ]]; then
      CONTROL_LOCK_HELD=0
      return 0
    fi
    if [[ "${owner_state}" == "3" ]]; then
      CONTROL_LOCK_HELD=0
      return 2
    fi
  done
  return 2
}

cleanup_control_lock() {
  local original_status=$?
  if [[ "${CONTROL_LOCK_HELD}" == "1" ]]; then
    release_control_lock || true
  fi
  return "${original_status}"
}

handle_signal() {
  local status="$1"
  trap - INT TERM
  exit "${status}"
}

trap cleanup_control_lock EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

sync_analysis_runtime() {
  local local_stage
  local hashes
  local remote_hashes
  git -C "${ROOT_DIR}" cat-file -e "${ANALYSIS_COMMIT}^{commit}"
  if ! git -C "${ROOT_DIR}" diff --quiet "${ANALYSIS_COMMIT}" -- "${FILES[@]}"; then
    echo "analysis files differ from ANALYSIS_COMMIT=${ANALYSIS_COMMIT}" >&2
    return 1
  fi
  local_stage="$(mktemp -d)"
  hashes="$(mktemp)"
  remote_hashes="$(mktemp)"
  git -C "${ROOT_DIR}" archive "${ANALYSIS_COMMIT}" "${FILES[@]}" | tar -x -C "${local_stage}"
  (cd "${local_stage}" && sha256sum "${FILES[@]}") > "${hashes}"

  if "${SSH_BIN}" "${REMOTE}" "test -d '${ANALYSIS_RELEASE}'"; then
    "${SSH_BIN}" "${REMOTE}" bash -s -- "${ANALYSIS_RELEASE}" "${ANALYSIS_COMMIT}" <<'REMOTE_VERIFY'
set -euo pipefail
release="$1"
commit="$2"
test "$(cat "${release}/DEPLOYED_COMMIT")" = "${commit}"
cd "${release}"
sha256sum -c .expected.sha256
REMOTE_VERIFY
    "${SSH_BIN}" "${REMOTE}" \
      "cd '${ANALYSIS_RELEASE}' && sha256sum ${FILES[*]}" > "${remote_hashes}"
    diff -u "${hashes}" "${remote_hashes}"
    log "verified analysis runtime ${ANALYSIS_RELEASE}"
    rm -r -- "${local_stage}"
    rm -- "${hashes}"
    rm -- "${remote_hashes}"
    return 0
  fi

  "${SSH_BIN}" "${REMOTE}" bash -s -- "${ANALYSIS_STAGE}" "$(dirname "${ANALYSIS_RELEASE}")" <<'REMOTE_STAGE'
set -euo pipefail
stage="$1"
deploy_root="$2"
mkdir -p "${deploy_root}"
test ! -e "${stage}"
mkdir "${stage}"
REMOTE_STAGE
  tar -C "${local_stage}" -cf - "${FILES[@]}" | \
    "${SSH_BIN}" "${REMOTE}" "tar -C '${ANALYSIS_STAGE}' -xf -"
  "${SSH_BIN}" "${REMOTE}" "tee '${ANALYSIS_STAGE}/.expected.sha256' >/dev/null" < "${hashes}"
  "${SSH_BIN}" "${REMOTE}" bash -s -- \
    "${ANALYSIS_STAGE}" "${ANALYSIS_RELEASE}" "${ANALYSIS_COMMIT}" <<'REMOTE_PUBLISH'
set -euo pipefail
stage="$1"
release="$2"
commit="$3"
cd "${stage}"
sha256sum -c .expected.sha256
printf '%s\n' "${commit}" > DEPLOYED_COMMIT
mv -- "${stage}" "${release}"
cd "${release}"
sha256sum -c .expected.sha256
REMOTE_PUBLISH
  "${SSH_BIN}" "${REMOTE}" \
    "cd '${ANALYSIS_RELEASE}' && sha256sum ${FILES[*]}" > "${remote_hashes}"
  diff -u "${hashes}" "${remote_hashes}"
  log "published analysis runtime ${ANALYSIS_RELEASE}"
  rm -r -- "${local_stage}"
  rm -- "${hashes}"
  rm -- "${remote_hashes}"
}

wait_for_final_acceptance() {
  local run_dir="$1"
  local output
  local status
  local started_at="${SECONDS}"
  while true; do
    if output="$("${SSH_BIN}" "${REMOTE}" \
      "${REMOTE_PYTHON}" "${ANALYSIS_RELEASE}/scripts/check_final_acceptance.py" \
      "${run_dir}")"; then
      log "final acceptance ready ${run_dir} ${output}"
      return 0
    else
      status=$?
    fi
    if [[ "${status}" == "2" ]]; then
      log "final acceptance failed ${run_dir} ${output}"
      return 1
    fi
    if [[ "${status}" != "1" ]]; then
      log "acceptance check transport/runtime failure status=${status} ${run_dir} ${output}"
      return 2
    fi
    if ((SECONDS - started_at >= WAIT_TIMEOUT_SECONDS)); then
      log "acceptance wait timed out after ${WAIT_TIMEOUT_SECONDS}s ${run_dir} ${output}"
      return 1
    fi
    log "waiting for final acceptance ${run_dir} ${output}"
    sleep "${POLL_SECONDS}"
  done
}

ensure_external_regrade() {
  local run_dir="$1"
  local state
  if "${SSH_BIN}" "${REMOTE}" python3 - "${run_dir}" "${EXTERNAL_VIEW}" <<'REMOTE_VIEW'
import sys
from pathlib import Path

run = Path(sys.argv[1])
view = sys.argv[2]
states = []
for step in (50, 100, 200):
    root = run / f"eval_step_{step}_n8" / view
    states.append(all((root / name).is_file() for name in ("summary.json", "input_hashes.sha256", "output_hashes.sha256")))
if all(states):
    raise SystemExit(0)
if any((run / f"eval_step_{step}_n8" / view).exists() for step in (50, 100, 200)):
    raise SystemExit(2)
raise SystemExit(1)
REMOTE_VIEW
  then
    log "external regrade already complete ${run_dir} view=${EXTERNAL_VIEW}"
    return 0
  else
    state=$?
  fi
  if [[ "${state}" == "2" ]]; then
    log "partial external regrade exists; refusing overwrite ${run_dir}"
    return 1
  fi
  "${SSH_BIN}" "${REMOTE}" bash -s -- \
    "${REMOTE_PYTHON}" "${ANALYSIS_RELEASE}/scripts/regrade_opd_eval_external.py" \
    "${run_dir}" "${GRADER_SOURCE}" "${EXTERNAL_VIEW}" <<'REMOTE_REGRADE'
set -euo pipefail
python="$1"
script="$2"
run="$3"
grader="$4"
view="$5"
"${python}" "${script}" \
  --run-dir "${run}" \
  --grader-source "${grader}" \
  --steps 50,100,200 \
  --view-name "${view}"
REMOTE_REGRADE
}

write_comparison() {
  local view="$1"
  local output="$2"
  local stage="${output}.stage_${OWNER_TOKEN}"
  "${SSH_BIN}" "${REMOTE}" bash -s -- \
    "${REMOTE_PYTHON}" "${ANALYSIS_RELEASE}/scripts/compare_paired_opd_evals.py" \
    "${LEFT_RUN}" "${RIGHT_RUN}" "${view}" "${stage}" <<'REMOTE_COMPARE'
set -euo pipefail
python="$1"
script="$2"
left="$3"
right="$4"
view="$5"
stage="$6"
mkdir -p "$(dirname "${stage}")"
test ! -e "${stage}"
"${python}" "${script}" \
  --left-run "${left}" \
  --right-run "${right}" \
  --steps 50,100,200 \
  --view "${view}" \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260712 \
  --output-json "${stage}" >/dev/null
final="${stage%.stage_*}"
if [[ -f "${final}" ]]; then
  cmp -s "${stage}" "${final}"
  rm -- "${stage}"
else
  mv -- "${stage}" "${final}"
fi
REMOTE_COMPARE
}

ensure_diagnostics() {
  local run_dir="$1"
  local label="$2"
  local output="$3"
  local stage="${output}.stage_${OWNER_TOKEN}"
  local state
  local local_stage
  if "${SSH_BIN}" "${REMOTE}" bash -s -- "${output}" <<'REMOTE_DIAGNOSTIC_STATE'
set -euo pipefail
output="$1"
required=(scalar_alignment.png scalar_credit.png scalar_optimization.png position_heatmaps.png entropy_segments.png diagnostics.html)
if [[ -e "${output}" ]]; then
  if [[ ! -d "${output}" ]]; then
    exit 2
  fi
  for name in "${required[@]}"; do
    if [[ ! -s "${output}/${name}" ]]; then
      exit 2
    fi
  done
  exit 0
fi
exit 1
REMOTE_DIAGNOSTIC_STATE
  then
    return 0
  else
    state=$?
  fi
  if [[ "${state}" == "2" ]]; then
    log "partial diagnostic output exists; refusing overwrite ${output}"
    return 1
  fi
  if [[ "${state}" != "1" ]]; then
    return "${state}"
  fi

  local_stage="$(mktemp -d)"
  mkdir -p "${local_stage}/run" "${local_stage}/runtime"
  git -C "${ROOT_DIR}" archive "${ANALYSIS_COMMIT}" \
    scripts/analyze_single_opd_diagnostics.py \
    scripts/analyze_block10_collapse_diagnostics.py \
    opd_ext/__init__.py opd_ext/analysis.py opd_ext/diagnostics.py | \
    tar -x -C "${local_stage}/runtime"
  "${SCP_BIN}" -r "${REMOTE}:${run_dir}/diagnostics" "${local_stage}/run/diagnostics"
  "${LOCAL_PLOT_PYTHON}" \
    "${local_stage}/runtime/scripts/analyze_single_opd_diagnostics.py" \
    --run-dir "${local_stage}/run" \
    --output-dir "${local_stage}/output" \
    --label "${label}"
  "${SSH_BIN}" "${REMOTE}" "test ! -e '${stage}'"
  "${SSH_BIN}" "${REMOTE}" "mkdir -p '$(dirname "${stage}")'"
  "${SCP_BIN}" -r "${local_stage}/output" "${REMOTE}:${stage}"
  "${SSH_BIN}" "${REMOTE}" bash -s -- "${output}" "${stage}" <<'REMOTE_DIAGNOSTICS'
set -euo pipefail
output="$1"
stage="$2"
required=(scalar_alignment.png scalar_credit.png scalar_optimization.png position_heatmaps.png entropy_segments.png diagnostics.html)
for name in "${required[@]}"; do test -s "${stage}/${name}"; done
mkdir -p "$(dirname "${output}")"
mv -- "${stage}" "${output}"
REMOTE_DIAGNOSTICS
  rm -r -- "${local_stage}"
}

write_report() {
  local external_json="${OUTPUT_ROOT}/comparisons/external.json"
  local builtin_json="${OUTPUT_ROOT}/comparisons/builtin.json"
  local output="${OUTPUT_ROOT}/report.html"
  local stage="${output}.stage_${OWNER_TOKEN}"
  "${SSH_BIN}" "${REMOTE}" bash -s -- \
    "${REMOTE_PYTHON}" "${ANALYSIS_RELEASE}/scripts/build_paired_validation_report.py" \
    "${external_json}" "${builtin_json}" "${stage}" \
    "${OUTPUT_ROOT}/assets/token_opd" "${OUTPUT_ROOT}/assets/block3_mean" \
    "${REPORT_TITLE_B64}" <<'REMOTE_REPORT'
set -euo pipefail
python="$1"
script="$2"
external="$3"
builtin="$4"
stage="$5"
token_diagnostics="$6"
block_diagnostics="$7"
title="$(printf '%s' "$8" | base64 --decode)"
test ! -e "${stage}"
"${python}" "${script}" \
  --external-json "${external}" \
  --builtin-json "${builtin}" \
  --output-html "${stage}" \
  --token-diagnostics "${token_diagnostics}" \
  --block-diagnostics "${block_diagnostics}" \
  --title "${title}"
final="${stage%.stage_*}"
if [[ -f "${final}" ]]; then
  cmp -s "${stage}" "${final}"
  rm -- "${stage}"
else
  mv -- "${stage}" "${final}"
fi
REMOTE_REPORT
}

write_finalization_hashes() {
  "${SSH_BIN}" "${REMOTE}" bash -s -- \
    "${OUTPUT_ROOT}" "${ANALYSIS_COMMIT}" "${LEFT_RUN}" "${RIGHT_RUN}" \
    "${EXTERNAL_VIEW}" <<'REMOTE_HASHES'
set -euo pipefail
root="$1"
commit="$2"
left="$3"
right="$4"
external_view="$5"
stage="${root}/finalization_hashes.sha256.stage_$$"
final="${root}/finalization_hashes.sha256"
commit_stage="${root}/ANALYSIS_COMMIT.stage_$$"
commit_final="${root}/ANALYSIS_COMMIT"
printf '%s\n' "${commit}" > "${commit_stage}"
if [[ -f "${commit_final}" ]]; then
  cmp -s "${commit_stage}" "${commit_final}"
  rm -- "${commit_stage}"
else
  mv -- "${commit_stage}" "${commit_final}"
fi
paths="${root}/.finalization_paths_$$"
{
  find "${root}/comparisons" "${root}/assets" -type f -print0
  printf '%s\0' "${root}/report.html" "${commit_final}"
  for run in "${left}" "${right}"; do
    find "${run}" -maxdepth 1 -type f -print0
    find "${run}/diagnostics" -type f -print0
    if [[ -d "${run}/grading" ]]; then find "${run}/grading" -type f -print0; fi
    if [[ -d "${run}/checkpoints" ]]; then
      find "${run}/checkpoints" -maxdepth 2 -type f \
        \( -name data.pt -o -name latest_checkpointed_iteration.txt \) -print0
    fi
    for step in 50 100 200; do
      eval_root="${run}/eval_step_${step}_n8"
      find "${eval_root}" -maxdepth 1 -type f -print0
      find "${eval_root}/outputs" -type f -print0
      find "${eval_root}/${external_view}" -type f -print0
      test -s "${eval_root}/${external_view}/input_hashes.sha256"
      test -s "${eval_root}/${external_view}/output_hashes.sha256"
    done
    test -s "${run}/acceptance.json"
  done
} | sort -zu > "${paths}"
xargs -0 sha256sum < "${paths}" > "${stage}"
rm -- "${paths}"
if [[ -f "${final}" ]]; then
  cmp -s "${stage}" "${final}"
  rm -- "${stage}"
else
  mv -- "${stage}" "${final}"
fi
REMOTE_HASHES
}

acquire_control_lock
sync_analysis_runtime
wait_for_final_acceptance "${LEFT_RUN}"
wait_for_final_acceptance "${RIGHT_RUN}"
ensure_external_regrade "${LEFT_RUN}"
ensure_external_regrade "${RIGHT_RUN}"
write_comparison "${EXTERNAL_VIEW}" "${OUTPUT_ROOT}/comparisons/external.json"
write_comparison outputs "${OUTPUT_ROOT}/comparisons/builtin.json"
ensure_diagnostics "${LEFT_RUN}" token_opd "${OUTPUT_ROOT}/assets/token_opd"
ensure_diagnostics "${RIGHT_RUN}" block3_mean "${OUTPUT_ROOT}/assets/block3_mean"
write_report
write_finalization_hashes
log "paired OPD finalization complete report=${OUTPUT_ROOT}/report.html"
