#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-ml2}"
REMOTE_ROOT="${REMOTE_ROOT:-/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd}"
RUN_ROOT="${RUN_ROOT:-${REMOTE_ROOT}/runs/20260709_block10_dapo17k_paper_qwen3}"

ssh "${REMOTE}" "set -euo pipefail
mkdir -p '${RUN_ROOT}/pipeline_logs'
if [[ -f '${RUN_ROOT}/pipeline.pid' ]] && ps -p \"\$(cat '${RUN_ROOT}/pipeline.pid')\" >/dev/null 2>&1; then
  echo \"already_running pid=\$(cat '${RUN_ROOT}/pipeline.pid') log=${RUN_ROOT}/pipeline_logs/pipeline.nohup.log\"
  exit 0
fi
nohup bash '${REMOTE_ROOT}/scripts/run_block10_paper_pipeline_on_ml2.sh' > '${RUN_ROOT}/pipeline_logs/pipeline.nohup.log' 2>&1 &
echo \$! > '${RUN_ROOT}/pipeline.pid'
echo \"pid=\$(cat '${RUN_ROOT}/pipeline.pid') log=${RUN_ROOT}/pipeline_logs/pipeline.nohup.log\"
"
