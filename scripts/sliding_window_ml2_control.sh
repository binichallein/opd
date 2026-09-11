#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  echo "usage: $0 {random3|sliding3} [sync|probe1|probe2|probe-audit|formal|status|eval [50|100|200]|audit-checkpoints|audit] (default: status)" >&2
  exit 2
}

VARIANT="${1:-}"
case "${VARIANT}" in
  random3|sliding3) shift ;;
  *) usage ;;
esac
ACTION="${1:-status}"
case "${ACTION}" in
  eval)
    [[ $# -le 2 ]] || usage
    case "${2:-200}" in 50|100|200) ;; *) usage ;; esac
    ;;
  sync|probe1|probe2|probe-audit|formal|status|audit-checkpoints|audit)
    [[ $# -le 1 ]] || usage
    ;;
  *) usage ;;
esac

# Keep the new experiment off the historical paired runs and their caches.
export REMOTE="ml2"
export HOST_TAG="ml2"
export ASSET_ROOT="/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd"
export SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"
if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_COMMIT must be an immutable full Git commit hash" >&2
  exit 2
fi
export RUNTIME_ROOT="${ASSET_ROOT}/deployments/${SOURCE_COMMIT}"
export REMOTE_ROOT="${RUNTIME_ROOT}"
export AUDIT_SCRIPT_OVERRIDE="${RUNTIME_ROOT}/scripts/audit_window_control.py"
export DATE_TAG="20260911v2r1"
export RUN_TAG="sliding_window"
export VARIANT
export PROJECT_NAME="opd_sliding_window"
export EXP_PREFIX="${VARIANT}-sliding-window-ml2"
# Ray appends a session timestamp and socket name; Linux allows only 107 bytes.
case "${VARIANT}" in
  random3) export CACHE_ROOT="/limx_embap/tos/wr/3r" ;;
  sliding3) export CACHE_ROOT="/limx_embap/tos/wr/3s" ;;
esac
export ENV_SEED=21
export OPD_WINDOW_SEED="$((910000 + ENV_SEED))"
export ROLLOUT_GPU_MEMORY_UTILIZATION="0.6"
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="1"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="4"
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="1"
unset OPD_DIAG_OUTPUT_DIR

# The adapter runs the original audit before checking window state and metadata.
exec bash "${ROOT_DIR}/scripts/block3_replication_control.sh" "$@"
