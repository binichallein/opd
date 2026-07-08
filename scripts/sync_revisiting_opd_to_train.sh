#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-train}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data/cpfs/Yaleon/opd}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"
bash scripts/setup_revisiting_opd.sh >/dev/null

ssh "${REMOTE}" "mkdir -p '${REMOTE_ROOT}'"

tar \
  --exclude='./.git' \
  --exclude='*/.git' \
  --exclude='./.pytest_cache' \
  --exclude='*/__pycache__' \
  --exclude='./runs' \
  --exclude='./outputs' \
  --exclude='./checkpoints' \
  --exclude='./models' \
  --exclude='./data/raw' \
  --exclude='./data/processed' \
  --exclude='./data/cache' \
  -C "${ROOT_DIR}" \
  -czf - . | ssh "${REMOTE}" "tar -xzf - -C '${REMOTE_ROOT}'"

ssh "${REMOTE}" "set -e
cd '${REMOTE_ROOT}'
grep -n 'aggregate_blockwise_policy_inputs\\|opd_block_size' \
  external/revisiting_opd/verl/trainer/ppo/core_algos.py \
  external/revisiting_opd/verl/workers/actor/dp_actor.py \
  external/revisiting_opd/verl/trainer/config/ppo_trainer.yaml | sed -n '1,40p'
du -sh '${REMOTE_ROOT}'
"
