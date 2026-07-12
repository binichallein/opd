#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-}"
case "${VARIANT}" in
  token_opd|block3_mean)
    shift
    ;;
  *)
    echo "usage: $0 {token_opd|block3_mean} {sync|probe1|probe2|probe-audit|formal|status|eval [step]|audit-checkpoints|audit}" >&2
    exit 2
    ;;
esac
export VARIANT="${VARIANT}"

export REMOTE="train"
export HOST_TAG="train"
export ASSET_ROOT="/mnt/data/cpfs/Yaleon/opd"
export SOURCE_COMMIT="${DEEPSEEK_JUSTRL_RUNTIME_COMMIT:-87274bfb3d1956a385fd0f43591319cb723577d2}"
export RUNTIME_ROOT="${ASSET_ROOT}/deployments/${SOURCE_COMMIT}"
export REMOTE_ROOT="${RUNTIME_ROOT}"
export VENV="/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/venv"
export HF_HOME_DIR="/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/hf_home"
export STUDENT_MODEL="${ASSET_ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B"
export MATH_TEACHER="${ASSET_ROOT}/models/JustRL-DeepSeek-1.5B"
export EXPECTED_STUDENT_MODEL_SUFFIX="DeepSeek-R1-Distill-Qwen-1.5B"
export EXPECTED_TEACHER_MODEL_SUFFIX="JustRL-DeepSeek-1.5B"
export EXPECTED_STUDENT_REVISION="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
export EXPECTED_TEACHER_REVISION="0637e4096c789c67f9eecbe8355e0bdeddede1c2"
export DATA_DIR="${ASSET_ROOT}/data/math_opd_dapo17k_hf_full_eval4"
export DATE_TAG="20260712v1"
export RUN_TAG="deepseek_justrl_pair"
export PROJECT_NAME="opd_deepseek_justrl_pair"
export EXP_PREFIX="deepseek-justrl-${VARIANT}-train"
export CACHE_ROOT="/mnt/data/cpfs/Yaleon/djr/0712v1/${VARIANT}"
export MIN_TOS_AVAILABLE_BYTES="1000000000000"
export ROLLOUT_GPU_MEMORY_UTILIZATION="0.6"
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="1"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="4"
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="1"
export BASELINE_ALIGNMENT="DeepSeek-R1-Distill-Qwen-1.5B student and its post-RL JustRL-DeepSeek-1.5B teacher; identical tokenizer and architecture; DAPO-Math-17K"

exec bash "${ROOT_DIR}/scripts/block3_replication_control.sh" "$@"
