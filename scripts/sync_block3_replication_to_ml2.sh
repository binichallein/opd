#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML2_ROOT="${ML2_ROOT:-/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd}"
SOURCE_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
DEPLOY_ROOT="${ML2_ROOT}/deployments"
RELEASE_DIR="${DEPLOY_ROOT}/${SOURCE_COMMIT}"
SYNC_STAGE="${DEPLOY_ROOT}/.stage_${SOURCE_COMMIT}_$$"

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
)

while IFS= read -r path; do
  FILES+=("external/revisiting_opd/${path}")
done < <(git -C "${ROOT_DIR}/external/revisiting_opd" ls-files)

for path in "${FILES[@]}"; do
  test -f "${ROOT_DIR}/${path}"
done

local_hashes="$(mktemp)"
remote_hashes="$(mktemp)"
trap 'rm -f "${local_hashes}" "${remote_hashes}"' EXIT

(cd "${ROOT_DIR}" && sha256sum "${FILES[@]}") > "${local_hashes}"

if ssh ml2 "test -d '${RELEASE_DIR}'"; then
  ssh ml2 "set -euo pipefail
    test \"\$(cat '${RELEASE_DIR}/DEPLOYED_COMMIT')\" = '${SOURCE_COMMIT}'
    cd '${RELEASE_DIR}'
    sha256sum -c .expected.sha256"
  ssh ml2 "cd '${RELEASE_DIR}' && sha256sum ${FILES[*]}" > "${remote_hashes}"
  diff -u "${local_hashes}" "${remote_hashes}"
  cat "${remote_hashes}"
  exit 0
fi

ssh ml2 "set -euo pipefail
  mkdir -p '${DEPLOY_ROOT}'
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
  printf '%s\n' '${SOURCE_COMMIT}' > DEPLOYED_COMMIT
  mv -- '${SYNC_STAGE}' '${RELEASE_DIR}'
  temporary_link='${DEPLOY_ROOT}/.current_${SOURCE_COMMIT}'
  ln -s '${RELEASE_DIR}' \"\${temporary_link}\"
  mv -Tf \"\${temporary_link}\" '${DEPLOY_ROOT}/current'
  cd '${RELEASE_DIR}'
  sha256sum -c .expected.sha256
  test \"\$(cat DEPLOYED_COMMIT)\" = '${SOURCE_COMMIT}'"
ssh ml2 "cd '${RELEASE_DIR}' && sha256sum ${FILES[*]}" > "${remote_hashes}"
diff -u "${local_hashes}" "${remote_hashes}"
cat "${remote_hashes}"
