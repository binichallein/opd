#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE_DIR="${ROOT_DIR}/external/revisiting_opd"
PATCH_FILE="${ROOT_DIR}/patches/revisiting_opd/blockwise_sampled_opd.patch"

cd "${ROOT_DIR}"
git submodule update --init --recursive external/revisiting_opd

cd "${SUBMODULE_DIR}"
APPLY_OPTS=(--ignore-space-change --whitespace=nowarn)

if git apply "${APPLY_OPTS[@]}" --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "blockwise sampled OPD patch is already applied"
else
  git apply "${APPLY_OPTS[@]}" --check "${PATCH_FILE}"
  git apply "${APPLY_OPTS[@]}" "${PATCH_FILE}"
  echo "applied blockwise sampled OPD patch"
fi

git diff --stat
