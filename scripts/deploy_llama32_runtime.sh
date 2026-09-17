#!/usr/bin/env bash
set -euo pipefail

# Staging is CPU/file-copy only. Never changes deployments/current or existing jobs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/deployments
COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
RELEASE="${TARGET}/${COMMIT}"
STAGE="${TARGET}/.llama32_stage_${COMMIT}_$$"
SUBMODULE="${ROOT}/external/revisiting_opd"
git -C "${ROOT}" diff --quiet HEAD -- . ':!external/revisiting_opd'
test -z "$(git -C "${ROOT}" ls-files --others --exclude-standard)"
test "$(git -C "${SUBMODULE}" rev-parse HEAD)" = "$(git -C "${ROOT}" ls-tree HEAD external/revisiting_opd | awk '{print $3}')"
git -C "${SUBMODULE}" apply --ignore-space-change --whitespace=nowarn --reverse --check \
    "${ROOT}/patches/revisiting_opd/blockwise_sampled_opd.patch"
(cd "${SUBMODULE}" && sha256sum --quiet -c "${ROOT}/manifests/revisiting_opd_runtime.sha256")

FILES="$(mktemp)"
HASHES="$(mktemp)"
trap 'rm -f "${FILES}" "${HASHES}"' EXIT
git -C "${ROOT}" ls-files -z -- . ':!external/revisiting_opd' > "${FILES}"
while IFS= read -r -d '' file; do
    printf 'external/revisiting_opd/%s\0' "${file}" >> "${FILES}"
done < <(git -C "${SUBMODULE}" ls-files -z)
(cd "${ROOT}" && xargs -0 sha256sum < "${FILES}") > "${HASHES}"

ssh ml2 "set -euo pipefail
  test ! -e '${RELEASE}'
  mkdir '${STAGE}'"
tar -C "${ROOT}" --null -T "${FILES}" -cf - | ssh ml2 "tar -C '${STAGE}' -xf -"
scp "${HASHES}" "ml2:${STAGE}/.expected.sha256"
ssh ml2 "set -euo pipefail
  cd '${STAGE}'
  sha256sum --quiet -c .expected.sha256
  printf '%s\n' '${COMMIT}' > DEPLOYED_COMMIT
  test ! -e '${RELEASE}'
  mv -T '${STAGE}' '${RELEASE}'
  echo 'Immutable isolated release: ${RELEASE}; no symlinks or GPU jobs changed'"
