#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-train}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data/cpfs/Yaleon/opd}"
RUN_ROOT="${RUN_ROOT:-${REMOTE_ROOT}/runs/20260708_block3_dapo17k_paper_qwen3}"
VARIANT="${VARIANT:-token_opd}"
VENV="${VENV:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/venv}"
HF_HOME_DIR="${HF_HOME_DIR:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/hf_home}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/tmp/opd_block3_dapo17k}"
SOURCE_COMMIT="${SOURCE_COMMIT:-unknown}"
SUBMODULE_BASE_COMMIT="${SUBMODULE_BASE_COMMIT:-f32f284f25bae5b16d2d44ee336b52851dccc736}"

PROJECT_NAME="${PROJECT_NAME:-opd_block3_dapo17k}"
EXP_NAME="${EXP_NAME:-${VARIANT}-qwen3-dapo17k-paper}"

STUDENT_MODEL="${STUDENT_MODEL:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-1.7B-Base}"
MATH_TEACHER="${MATH_TEACHER:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-4B-Base-GRPO}"
DATA_DIR="${DATA_DIR:-${REMOTE_ROOT}/data/math_opd_dapo17k_hf_full_eval4}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.parquet}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/test.parquet}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
ENV_SEED="${ENV_SEED:-21}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_GROUP_SIZE="${ROLLOUT_GROUP_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-1000000}"
VAL_N="${VAL_N:-1}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console']}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-18432}"
OPD_DIAGNOSTICS="${OPD_DIAGNOSTICS:-false}"
OPD_DIAG_INTERVAL="${OPD_DIAG_INTERVAL:-5}"
OPD_DIAG_TOPK="${OPD_DIAG_TOPK:-16}"
OPD_DIAG_POSITION_BIN="${OPD_DIAG_POSITION_BIN:-128}"
OPD_DIAG_POSITION_STRIDE="${OPD_DIAG_POSITION_STRIDE:-1}"
OPD_DIAG_SIGN_EPS="${OPD_DIAG_SIGN_EPS:-1e-4}"
DIAGNOSTIC_SAVE_STEPS="${DIAGNOSTIC_SAVE_STEPS:-40,50,60,80,100,200}"

RUN_DIR="${RUN_ROOT}/${VARIANT}"
CKPTS_DIR="${RUN_DIR}/checkpoints"
LOG_DIR="${RUN_DIR}/logs"
OPD_DIAG_OUTPUT_DIR="${OPD_DIAG_OUTPUT_DIR:-${RUN_DIR}/diagnostics}"

ssh "${REMOTE}" "set -euo pipefail
test -d '${REMOTE_ROOT}'
test -f '${REMOTE_ROOT}/scripts/run_revisiting_sampled_block_opd_math.sh'
test -d '${STUDENT_MODEL}'
test -d '${MATH_TEACHER}'
test -x '${VENV}/bin/python'
test -f '${TRAIN_DATA}'
test -f '${VAL_DATA}'
test \$(cat '${REMOTE_ROOT}/external/revisiting_opd.UPSTREAM_COMMIT') = '${SUBMODULE_BASE_COMMIT}'
if git -C '${REMOTE_ROOT}' rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  bash '${REMOTE_ROOT}/scripts/setup_revisiting_opd.sh' >/dev/null
fi
mkdir -p '${RUN_DIR}' '${CKPTS_DIR}' '${LOG_DIR}' '${RUN_DIR}/summaries' \
  '${LOCAL_CACHE_ROOT}/tmp' '${LOCAL_CACHE_ROOT}/vllm_cache' \
  '${LOCAL_CACHE_ROOT}/torchinductor' '${LOCAL_CACHE_ROOT}/triton' \
  '${LOCAL_CACHE_ROOT}/cuda_cache' '${LOCAL_CACHE_ROOT}/outlines'
(cd '${REMOTE_ROOT}/external/revisiting_opd' && \
  sha256sum -c '${REMOTE_ROOT}/manifests/revisiting_opd_runtime.sha256') \
  > '${RUN_DIR}/revisiting_opd_manifest_check.txt'
cp '${DATA_DIR}/manifest.json' '${RUN_DIR}/data_manifest.json'
sha256sum \
  '${REMOTE_ROOT}/scripts/launch_revisiting_block_opd_formal_train.sh' \
  '${REMOTE_ROOT}/scripts/run_revisiting_sampled_block_opd_math.sh' \
  '${REMOTE_ROOT}/scripts/setup_revisiting_opd.sh' \
  '${REMOTE_ROOT}/patches/revisiting_opd/blockwise_sampled_opd.patch' \
  '${REMOTE_ROOT}/external/revisiting_opd.UPSTREAM_COMMIT' \
  '${REMOTE_ROOT}/manifests/revisiting_opd_runtime.sha256' \
  '${REMOTE_ROOT}/opd_ext/diagnostics.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/trainer/ppo/core_algos.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/trainer/ppo/ray_trainer_multitask.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/workers/actor/dp_actor.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/workers/fsdp_workers.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/trainer/config/ppo_trainer.yaml' \
  > '${RUN_DIR}/script_hashes.sha256'
{
  sha256sum '${TRAIN_DATA}' '${VAL_DATA}'
  find '${STUDENT_MODEL}' '${MATH_TEACHER}' -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name '*.bin' -o -name '*.json' -o -name '*.jinja' -o -name '*.txt' \) \
    -print0 | sort -z | xargs -0 sha256sum
} > '${RUN_DIR}/artifact_hashes.sha256'
{
  date
  hostname
  echo 'python=${VENV}/bin/python'
  '${VENV}/bin/python' --version || true
  '${VENV}/bin/python' -c 'import torch, transformers, vllm; print(\"torch=\" + str(torch.__version__)); print(\"transformers=\" + transformers.__version__); print(\"vllm=\" + vllm.__version__)' || true
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} > '${RUN_DIR}/env.txt'
cat > '${RUN_DIR}/run_card.json' <<JSON
{
  \"variant\": \"${VARIANT}\",
  \"source_commit\": \"${SOURCE_COMMIT}\",
  \"revisiting_opd_base_commit\": \"${SUBMODULE_BASE_COMMIT}\",
  \"project_name\": \"${PROJECT_NAME}\",
  \"experiment_name\": \"${EXP_NAME}\",
  \"student_model\": \"${STUDENT_MODEL}\",
  \"teacher_model\": \"${MATH_TEACHER}\",
  \"venv\": \"${VENV}\",
  \"train_data\": \"${TRAIN_DATA}\",
  \"val_data\": \"${VAL_DATA}\",
  \"seed\": ${ENV_SEED},
  \"data_seed\": ${ENV_SEED},
  \"rollout_seed\": ${ENV_SEED},
  \"environment_seed\": ${ENV_SEED},
  \"n_gpus_per_node\": ${N_GPUS_PER_NODE},
  \"ray_num_cpus\": ${RAY_NUM_CPUS},
  \"train_batch_size\": ${TRAIN_BATCH_SIZE},
  \"ppo_mini_batch_size\": ${PPO_MINI_BATCH_SIZE},
  \"rollout_group_size\": ${ROLLOUT_GROUP_SIZE},
  \"max_prompt_length\": ${MAX_PROMPT_LENGTH},
  \"max_response_length\": ${MAX_RESPONSE_LENGTH},
  \"learning_rate\": ${LEARNING_RATE},
  \"total_training_steps\": ${TOTAL_TRAINING_STEPS},
  \"save_freq\": ${SAVE_FREQ},
  \"test_freq\": ${TEST_FREQ},
  \"val_n\": ${VAL_N},
  \"rollout_gpu_memory_utilization\": ${ROLLOUT_GPU_MEMORY_UTILIZATION},
  \"rollout_max_num_batched_tokens\": ${ROLLOUT_MAX_NUM_BATCHED_TOKENS},
  \"opd_diagnostics\": ${OPD_DIAGNOSTICS},
  \"opd_diag_interval\": ${OPD_DIAG_INTERVAL},
  \"opd_diag_topk\": ${OPD_DIAG_TOPK},
  \"opd_diag_position_bin\": ${OPD_DIAG_POSITION_BIN},
  \"opd_diag_position_stride\": ${OPD_DIAG_POSITION_STRIDE},
  \"opd_diag_sign_epsilon\": ${OPD_DIAG_SIGN_EPS},
  \"diagnostic_save_steps\": \"${DIAGNOSTIC_SAVE_STEPS}\",
  \"diagnostic_output_dir\": \"${OPD_DIAG_OUTPUT_DIR}\",
  \"checkpoint_policy\": \"preserve all milestone checkpoints; no automatic deletion\",
  \"baseline_alignment\": \"Blockwise/Rethinking-aligned Qwen3-1.7B-Base student, Qwen3-4B-Base-GRPO teacher, and raw 1,791,700-row DAPO-Math-17K pool; Revisiting OPD is codebase only\"
}
JSON
cat > '${RUN_DIR}/command.sh' <<'CMD'
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PATH='${VENV}/bin':"\$PATH"
export PYTHONPATH='${REMOTE_ROOT}/external/revisiting_opd':"\${PYTHONPATH:-}"
export HF_HOME='${HF_HOME_DIR}'
export HF_HUB_CACHE='${HF_HOME_DIR}/hub'
export HF_DATASETS_CACHE='${HF_HOME_DIR}/datasets'
export TMPDIR='${LOCAL_CACHE_ROOT}/tmp'
export VLLM_CACHE_ROOT='${LOCAL_CACHE_ROOT}/vllm_cache'
export TORCHINDUCTOR_CACHE_DIR='${LOCAL_CACHE_ROOT}/torchinductor'
export TRITON_CACHE_DIR='${LOCAL_CACHE_ROOT}/triton'
export CUDA_CACHE_PATH='${LOCAL_CACHE_ROOT}/cuda_cache'
export OUTLINES_CACHE_DIR='${LOCAL_CACHE_ROOT}/outlines'
export PYTHONUNBUFFERED=1
cd '${REMOTE_ROOT}'
VARIANT='${VARIANT}' \
PROJECT_NAME='${PROJECT_NAME}' \
EXP_NAME='${EXP_NAME}' \
STUDENT_MODEL='${STUDENT_MODEL}' \
MATH_TEACHER='${MATH_TEACHER}' \
TRAIN_DATA='${TRAIN_DATA}' \
VAL_DATA='${VAL_DATA}' \
CKPTS_DIR='${CKPTS_DIR}' \
LOG_DIR='${LOG_DIR}' \
N_GPUS_PER_NODE='${N_GPUS_PER_NODE}' \
RAY_NUM_CPUS='${RAY_NUM_CPUS}' \
ENV_SEED='${ENV_SEED}' \
TRAIN_BATCH_SIZE='${TRAIN_BATCH_SIZE}' \
PPO_MINI_BATCH_SIZE='${PPO_MINI_BATCH_SIZE}' \
ROLLOUT_GROUP_SIZE='${ROLLOUT_GROUP_SIZE}' \
MAX_PROMPT_LENGTH='${MAX_PROMPT_LENGTH}' \
MAX_RESPONSE_LENGTH='${MAX_RESPONSE_LENGTH}' \
LEARNING_RATE='${LEARNING_RATE}' \
TOTAL_TRAINING_STEPS='${TOTAL_TRAINING_STEPS}' \
SAVE_FREQ='${SAVE_FREQ}' \
TEST_FREQ='${TEST_FREQ}' \
VAL_N='${VAL_N}' \
ROLLOUT_GPU_MEMORY_UTILIZATION='${ROLLOUT_GPU_MEMORY_UTILIZATION}' \
ROLLOUT_MAX_NUM_BATCHED_TOKENS='${ROLLOUT_MAX_NUM_BATCHED_TOKENS}' \
TRAINER_LOGGER=\"${TRAINER_LOGGER}\" \
OPD_DIAGNOSTICS='${OPD_DIAGNOSTICS}' \
OPD_DIAG_INTERVAL='${OPD_DIAG_INTERVAL}' \
OPD_DIAG_TOPK='${OPD_DIAG_TOPK}' \
OPD_DIAG_POSITION_BIN='${OPD_DIAG_POSITION_BIN}' \
OPD_DIAG_POSITION_STRIDE='${OPD_DIAG_POSITION_STRIDE}' \
OPD_DIAG_SIGN_EPS='${OPD_DIAG_SIGN_EPS}' \
DIAGNOSTIC_SAVE_STEPS='${DIAGNOSTIC_SAVE_STEPS}' \
OPD_DIAG_OUTPUT_DIR='${OPD_DIAG_OUTPUT_DIR}' \
bash scripts/run_revisiting_sampled_block_opd_math.sh
CMD
chmod +x '${RUN_DIR}/command.sh'
if [[ -f '${RUN_DIR}/train.pid' ]] && ps -p \"\$(cat '${RUN_DIR}/train.pid')\" >/dev/null 2>&1; then
  echo \"already_running pid=\$(cat '${RUN_DIR}/train.pid') run_dir=${RUN_DIR}\"
  exit 0
fi
nohup bash '${RUN_DIR}/command.sh' > '${LOG_DIR}/nohup.log' 2>&1 &
echo \$! > '${RUN_DIR}/train.pid'
echo \"pid=\$(cat '${RUN_DIR}/train.pid') run_dir=${RUN_DIR} log=${LOG_DIR}/nohup.log\"
"
