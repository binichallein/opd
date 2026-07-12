#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The paired token baseline must use the exact immutable runtime used by Block3.
export SOURCE_COMMIT="9b3b8b76bcdf02d0a4cfe4720cecb24bc7203977"
export DATE_TAG="20260712v1"
export VARIANT="token_opd"
export RUN_TAG="token_opd_replication"
export PROJECT_NAME="opd_token_replication"
export EXP_PREFIX="token-opd-replication-ml2"
export CACHE_ROOT="/limx_embap/tos/tor/0712v1"
export ROLLOUT_GPU_MEMORY_UTILIZATION="0.6"
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="1"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="4"
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="1"

case "${1:-status}" in
  sync)
    echo "immutable runtime is already deployed at commit ${SOURCE_COMMIT}"
    ;;
  *)
    exec bash "${ROOT_DIR}/scripts/block3_replication_control.sh" "$@"
    ;;
esac
