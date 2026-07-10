#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-train}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data/cpfs/Yaleon/opd}"
RUN_ROOT="${RUN_ROOT:-${REMOTE_ROOT}/runs/20260708_block3_dapo17k_paper_qwen3}"

ssh "${REMOTE}" "set -euo pipefail
mkdir -p '${RUN_ROOT}/pipeline_logs'
if [[ -f '${RUN_ROOT}/pipeline.pid' ]] && ps -p \"\$(cat '${RUN_ROOT}/pipeline.pid')\" >/dev/null 2>&1; then
  echo \"already_running pid=\$(cat '${RUN_ROOT}/pipeline.pid') log=${RUN_ROOT}/pipeline_logs/pipeline.nohup.log\"
  exit 0
fi
nohup bash '${REMOTE_ROOT}/scripts/run_block3_paper_pipeline_on_train.sh' > '${RUN_ROOT}/pipeline_logs/pipeline.nohup.log' 2>&1 &
echo \$! > '${RUN_ROOT}/pipeline.pid'
echo \"pid=\$(cat '${RUN_ROOT}/pipeline.pid') log=${RUN_ROOT}/pipeline_logs/pipeline.nohup.log\"
"
