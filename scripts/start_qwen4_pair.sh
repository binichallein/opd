#!/usr/bin/env bash
set -euo pipefail

# Run under server-side nohup only after an immutable deployment is verified.
RUNTIME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd
PYTHON=/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl/bin/python
COMMIT="$(<"${RUNTIME}/DEPLOYED_COMMIT")"
test "${RUNTIME}" = "${ROOT}/deployments/${COMMIT}"
cd "${RUNTIME}"
sha256sum --quiet -c .expected.sha256
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ -e "${ROOT}/models/Qwen3-4B-Base" ]]; then
    "${PYTHON}" scripts/prepare_qwen4_assets.py
else
    "${PYTHON}" scripts/prepare_qwen4_assets.py --download
fi
exec "${PYTHON}" scripts/run_qwen4_pair.py
