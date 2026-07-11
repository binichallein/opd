#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML2_ROOT="${ML2_ROOT:-/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd}"

FILES=(
  opd_ext/__init__.py
  opd_ext/analysis.py
  opd_ext/diagnostics.py
  patches/revisiting_opd/blockwise_sampled_opd.patch
  scripts/setup_revisiting_opd.sh
  scripts/run_revisiting_sampled_block_opd_math.sh
  scripts/launch_revisiting_block_opd_formal_train.sh
  scripts/launch_qwen3_math_eval.sh
  scripts/eval_qwen3_math_vllm.py
  scripts/audit_block10_run.py
  scripts/block3_replication_control.sh
  scripts/sync_block3_replication_to_ml2.sh
  scripts/analyze_block10_collapse_diagnostics.py
  scripts/analyze_single_opd_diagnostics.py
  external/revisiting_opd.UPSTREAM_COMMIT
  manifests/revisiting_opd_runtime.sha256
  external/revisiting_opd/verl/trainer/config/ppo_trainer.yaml
  external/revisiting_opd/verl/trainer/ppo/core_algos.py
  external/revisiting_opd/verl/trainer/ppo/ray_trainer_multitask.py
  external/revisiting_opd/verl/workers/actor/dp_actor.py
  external/revisiting_opd/verl/workers/actor/megatron_actor.py
  external/revisiting_opd/verl/workers/fsdp_workers.py
  external/revisiting_opd/verl/utils/reward_score/math.py
)

for path in "${FILES[@]}"; do
  test -f "${ROOT_DIR}/${path}"
done

ssh ml2 "mkdir -p '${ML2_ROOT}'"
tar -C "${ROOT_DIR}" -cf - "${FILES[@]}" | ssh ml2 "tar -C '${ML2_ROOT}' -xf -"

local_hashes="$(mktemp)"
remote_hashes="$(mktemp)"
trap 'rm -f "${local_hashes}" "${remote_hashes}"' EXIT

(cd "${ROOT_DIR}" && sha256sum "${FILES[@]}") > "${local_hashes}"
ssh ml2 "cd '${ML2_ROOT}' && sha256sum ${FILES[*]}" > "${remote_hashes}"
diff -u "${local_hashes}" "${remote_hashes}"
cat "${remote_hashes}"
