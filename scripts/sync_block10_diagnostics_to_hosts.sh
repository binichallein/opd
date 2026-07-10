#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ROOT="${TRAIN_ROOT:-/mnt/data/cpfs/Yaleon/opd}"
ML2_ROOT="${ML2_ROOT:-/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd}"

FILES=(
  opd_ext/__init__.py
  opd_ext/diagnostics.py
  patches/revisiting_opd/blockwise_sampled_opd.patch
  scripts/setup_revisiting_opd.sh
  scripts/run_revisiting_sampled_block_opd_math.sh
  scripts/launch_revisiting_block_opd_formal_train.sh
  scripts/launch_qwen3_math_eval.sh
  scripts/eval_qwen3_math_vllm.py
  external/revisiting_opd.UPSTREAM_COMMIT
  manifests/revisiting_opd_runtime.sha256
  external/revisiting_opd/verl/trainer/config/ppo_trainer.yaml
  external/revisiting_opd/verl/trainer/ppo/core_algos.py
  external/revisiting_opd/verl/trainer/ppo/ray_trainer_multitask.py
  external/revisiting_opd/verl/workers/actor/dp_actor.py
  external/revisiting_opd/verl/workers/actor/megatron_actor.py
  external/revisiting_opd/verl/workers/fsdp_workers.py
)

sync_host() {
  local host="$1"
  local remote_root="$2"
  ssh "${host}" "mkdir -p '${remote_root}/opd_ext' '${remote_root}/scripts' \
    '${remote_root}/patches/revisiting_opd' \
    '${remote_root}/external' \
    '${remote_root}/manifests' \
    '${remote_root}/external/revisiting_opd/verl/trainer/config' \
    '${remote_root}/external/revisiting_opd/verl/trainer/ppo' \
    '${remote_root}/external/revisiting_opd/verl/workers/actor' \
    '${remote_root}/external/revisiting_opd/verl/workers'"
  tar -C "${ROOT_DIR}" -cf - "${FILES[@]}" | ssh "${host}" "tar -C '${remote_root}' -xf -"
  ssh "${host}" "cd '${remote_root}' && sha256sum ${FILES[*]}"
}

sync_host train "${TRAIN_ROOT}"
sync_host ml2 "${ML2_ROOT}"
