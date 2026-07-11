#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML2_ROOT="${ML2_ROOT:-/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd}"
SOURCE_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
SYNC_STAGE="${ML2_ROOT}/.sync_staging_${SOURCE_COMMIT}"

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

local_hashes="$(mktemp)"
remote_hashes="$(mktemp)"
trap 'rm -f "${local_hashes}" "${remote_hashes}"' EXIT

(cd "${ROOT_DIR}" && sha256sum "${FILES[@]}") > "${local_hashes}"
ssh ml2 "set -euo pipefail
  mkdir -p '${ML2_ROOT}'
  if ps -eo cmd | grep -E '[m]ain_ppo_multitask|[r]aylet|[r]ay::|[V]LLMWorker' >/dev/null; then
    echo 'refusing source sync while training, Ray, or vLLM is active' >&2
    exit 1
  fi
  rm -rf -- '${SYNC_STAGE}'
  mkdir -p '${SYNC_STAGE}'"
tar -C "${ROOT_DIR}" -cf - "${FILES[@]}" | ssh ml2 "tar -C '${SYNC_STAGE}' -xf -"
ssh ml2 "tee '${SYNC_STAGE}/.expected.sha256' >/dev/null" < "${local_hashes}"
printf '%s\n' "${FILES[@]}" | ssh ml2 "tee '${SYNC_STAGE}/.files' >/dev/null"
ssh ml2 "set -euo pipefail
  cd '${SYNC_STAGE}'
  sha256sum -c .expected.sha256
  while IFS= read -r path; do
    install -D -- '${SYNC_STAGE}/'\"\${path}\" '${ML2_ROOT}/'\"\${path}\"
  done < .files
  cd '${ML2_ROOT}'
  sha256sum -c '${SYNC_STAGE}/.expected.sha256'
  rm -rf -- '${SYNC_STAGE}'"
ssh ml2 "cd '${ML2_ROOT}' && sha256sum ${FILES[*]}" > "${remote_hashes}"
diff -u "${local_hashes}" "${remote_hashes}"
cat "${remote_hashes}"
