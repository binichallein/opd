#!/usr/bin/env bash
set -euo pipefail

LOCAL_STAGE="${LOCAL_STAGE:-/home/tyf/paper/.cache/block10_ml2_sync}"
LOCAL_REPO="${LOCAL_REPO:-/home/tyf/paper/opd}"
REMOTE="${REMOTE:-ml2}"
REMOTE_ROOT="${REMOTE_ROOT:-/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd}"

TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/mnt/data/cpfs/Yaleon/opd/data/math_opd_dapo17k_hf_full_eval4}"
TRAIN_BASELINE_ROOT="${TRAIN_BASELINE_ROOT:-/mnt/data/cpfs/Yaleon/opd/runs/20260708_block3_dapo17k_paper_qwen3}"

LOG_DIR="${LOCAL_STAGE}/logs"
mkdir -p "${LOCAL_STAGE}" "${LOG_DIR}"
LOG="${LOG_DIR}/sync_$(date +%Y%m%d_%H%M%S).log"

run() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

pull_tar_dir() {
  local remote_parent="$1"
  local dir_name="$2"
  local local_parent="$3"
  echo "[$(date '+%F %T')] pull train:${remote_parent}/${dir_name} -> ${local_parent}/${dir_name}" | tee -a "${LOG}"
  ssh train "tar -C '${remote_parent}' -cf - '${dir_name}'" | tar -C "${local_parent}" -xf -
  echo "[$(date '+%F %T')] pulled ${dir_name}" | tee -a "${LOG}"
}

echo "[$(date '+%F %T')] sync start" | tee -a "${LOG}"

mkdir -p \
  "${LOCAL_STAGE}/models" \
  "${LOCAL_STAGE}/data" \
  "${LOCAL_STAGE}/runs/20260708_block3_dapo17k_paper_qwen3/token_opd/eval_step_200_n8/outputs" \
  "${LOCAL_STAGE}/runs/20260708_block3_dapo17k_paper_qwen3/block3_mean/eval_step_200_n8/outputs"

DATA_PARENT="$(dirname "${TRAIN_DATA_ROOT}")"
DATA_BASENAME="$(basename "${TRAIN_DATA_ROOT}")"

pull_tar_dir "${TRAIN_MODEL_ROOT}" "Qwen3-4B-Base-GRPO" "${LOCAL_STAGE}/models"

pull_tar_dir "${DATA_PARENT}" "${DATA_BASENAME}" "${LOCAL_STAGE}/data"

run scp -O \
  "train:${TRAIN_BASELINE_ROOT}/token_opd/eval_step_200_n8/outputs/summary.json" \
  "${LOCAL_STAGE}/runs/20260708_block3_dapo17k_paper_qwen3/token_opd/eval_step_200_n8/outputs/summary.json"

run scp -O \
  "train:${TRAIN_BASELINE_ROOT}/block3_mean/eval_step_200_n8/outputs/summary.json" \
  "${LOCAL_STAGE}/runs/20260708_block3_dapo17k_paper_qwen3/block3_mean/eval_step_200_n8/outputs/summary.json"

run ssh "${REMOTE}" "mkdir -p '${REMOTE_ROOT}'"

run rsync -a --partial --info=progress2 \
  "${LOCAL_REPO}/external" \
  "${LOCAL_REPO}/scripts" \
  "${LOCAL_REPO}/reports" \
  "${LOCAL_REPO}/docs" \
  "${LOCAL_REPO}/README.md" \
  "${REMOTE}:${REMOTE_ROOT}/"

run rsync -a --partial --info=progress2 \
  "${LOCAL_STAGE}/models/" \
  "${REMOTE}:${REMOTE_ROOT}/models/"

run rsync -a --partial --info=progress2 \
  "${LOCAL_STAGE}/data/" \
  "${REMOTE}:${REMOTE_ROOT}/data/"

run rsync -a --partial --info=progress2 \
  "${LOCAL_STAGE}/runs/" \
  "${REMOTE}:${REMOTE_ROOT}/runs/"

run ssh "${REMOTE}" "cd '${REMOTE_ROOT}' && \
  mkdir -p runs/20260709_block10_dapo17k_paper_qwen3 && \
  sha256sum \
    data/math_opd_dapo17k_hf_full_eval4/train.parquet \
    data/math_opd_dapo17k_hf_full_eval4/test.parquet \
    data/math_opd_dapo17k_hf_full_eval4/manifest.json \
    data/math_opd_dapo17k_hf_full_eval4/eval_jsonl/*.jsonl \
    models/Qwen3-4B-Base-GRPO/config.json \
    models/Qwen3-4B-Base-GRPO/*.safetensors \
    > runs/20260709_block10_dapo17k_paper_qwen3/synced_asset_hashes.sha256"

echo "[$(date '+%F %T')] sync complete" | tee -a "${LOG}"
echo "log=${LOG}"
