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
PREPARE_ONLY="${PREPARE_ONLY:-false}"
OPD_PROMPT_PROTOCOL="${OPD_PROMPT_PROTOCOL:-legacy}"
OPD_REQUEST_SEED_RULE="${OPD_REQUEST_SEED_RULE:-legacy}"
LOSSLESS_ROLLOUT_DIR="${LOSSLESS_ROLLOUT_DIR:-}"
ROLLOUT_ATTEMPT_ID="${ROLLOUT_ATTEMPT_ID:-}"
case "${PREPARE_ONLY}" in
  true|false) ;;
  *) echo "PREPARE_ONLY must be true or false" >&2; exit 2 ;;
esac

PROJECT_NAME="${PROJECT_NAME:-opd_block3_dapo17k}"
EXP_NAME="${EXP_NAME:-${VARIANT}-qwen3-dapo17k-paper}"

STUDENT_MODEL="${STUDENT_MODEL:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-1.7B-Base}"
MATH_TEACHER="${MATH_TEACHER:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-4B-Base-GRPO}"
STUDENT_MODEL_REVISION="${STUDENT_MODEL_REVISION:-}"
TEACHER_MODEL_REVISION="${TEACHER_MODEL_REVISION:-}"
BASELINE_ALIGNMENT="${BASELINE_ALIGNMENT:-Blockwise/Rethinking-aligned Qwen3-1.7B-Base student, Qwen3-4B-Base-GRPO teacher, and raw 1,791,700-row DAPO-Math-17K pool; Revisiting OPD is codebase only}"
DATA_DIR="${DATA_DIR:-${REMOTE_ROOT}/data/math_opd_dapo17k_hf_full_eval4}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.parquet}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/test.parquet}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
ENV_SEED="${ENV_SEED:-21}"
OPD_WINDOW_SEED="${OPD_WINDOW_SEED:-$((910000 + ENV_SEED))}"
OPD_WINDOW_MODE=fixed
OPD_BLOCK_SIZE=3
OPD_BLOCK_ADVANTAGE_MODE=mean
case "${VARIANT}" in
  random3|sliding3) OPD_WINDOW_MODE="${VARIANT%3}" ;;
  token_opd) OPD_BLOCK_SIZE=1; OPD_BLOCK_ADVANTAGE_MODE=sum ;;
  block3_sum) OPD_BLOCK_ADVANTAGE_MODE=sum ;;
  block3_mean) ;;
  block5_mean) OPD_BLOCK_SIZE=5 ;;
  block10_mean) OPD_BLOCK_SIZE=10 ;;
  block3_mixed_lam05) OPD_BLOCK_ADVANTAGE_MODE=mixed ;;
  *) echo "Unknown VARIANT=${VARIANT}" >&2; exit 2 ;;
esac
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
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
OPD_DIAGNOSTICS="${OPD_DIAGNOSTICS:-false}"
OPD_DIAG_INTERVAL="${OPD_DIAG_INTERVAL:-5}"
OPD_DIAG_TOPK="${OPD_DIAG_TOPK:-16}"
OPD_DIAG_POSITION_BIN="${OPD_DIAG_POSITION_BIN:-128}"
OPD_DIAG_POSITION_STRIDE="${OPD_DIAG_POSITION_STRIDE:-1}"
OPD_DIAG_SIGN_EPS="${OPD_DIAG_SIGN_EPS:-1e-4}"
DIAGNOSTIC_SAVE_STEPS="${DIAGNOSTIC_SAVE_STEPS:-40,50,60,80,100,200}"
STOP_AFTER_STEP="${STOP_AFTER_STEP:--1}"
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-true}"
EXPECTED_TRAIN_SHA256="${EXPECTED_TRAIN_SHA256:-}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

RUN_DIR="${RUN_ROOT}/${VARIANT}"
CKPTS_DIR="${RUN_DIR}/checkpoints"
LOG_DIR="${RUN_DIR}/logs"
OPD_DIAG_OUTPUT_DIR="${OPD_DIAG_OUTPUT_DIR:-${RUN_DIR}/diagnostics}"

run_on_target() {
  case "${LAUNCH_TRANSPORT:-ssh}" in
    ssh) ssh "${REMOTE}" "$1" ;;
    local)
      if [[ "${REMOTE}" != ml2 || "${PREPARE_ONLY}" != true ||
            ! "${REMOTE_ROOT}" =~ ^/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/deployments/[0-9a-f]{40}$ ]]; then
        echo 'local transport requires approved ml2 runtime and PREPARE_ONLY=true' >&2
        exit 2
      fi
      bash -c "$1"
      ;;
    *) echo 'Unknown LAUNCH_TRANSPORT' >&2; exit 2 ;;
  esac
}

run_on_target "set -euo pipefail
test -d '${REMOTE_ROOT}'
if [[ '${SOURCE_COMMIT}' != unknown ]]; then
  test \"\$(cat '${REMOTE_ROOT}/DEPLOYED_COMMIT')\" = '${SOURCE_COMMIT}'
fi
test -f '${REMOTE_ROOT}/scripts/run_revisiting_sampled_block_opd_math.sh'
test -d '${STUDENT_MODEL}'
test -d '${MATH_TEACHER}'
if [[ -n '${STUDENT_MODEL_REVISION}' ]]; then
  revision_file='${STUDENT_MODEL}/HF_REVISION'
  if [[ '${OPD_PROMPT_PROTOCOL}' == llama32_nonthinking_v1 || '${OPD_PROMPT_PROTOCOL}' == llama32_historical17_v1 || '${OPD_PROMPT_PROTOCOL}' == qwen3_historical17_v1 || '${OPD_PROMPT_PROTOCOL}' == qwen3_completion_boxed_v1 || '${OPD_PROMPT_PROTOCOL}' == qwen3_native_chat_no_thinking_boxed_v1 ]]; then revision_file='${STUDENT_MODEL}/SOURCE_REVISION'; fi
  test \"\$(cat \"\${revision_file}\")\" = '${STUDENT_MODEL_REVISION}'
fi
if [[ -n '${TEACHER_MODEL_REVISION}' ]]; then
  revision_file='${MATH_TEACHER}/HF_REVISION'
  if [[ '${OPD_PROMPT_PROTOCOL}' == llama32_nonthinking_v1 || '${OPD_PROMPT_PROTOCOL}' == llama32_historical17_v1 || '${OPD_PROMPT_PROTOCOL}' == qwen3_native_chat_no_thinking_boxed_v1 ]]; then revision_file='${MATH_TEACHER}/SOURCE_REVISION'; fi
  test \"\$(cat \"\${revision_file}\")\" = '${TEACHER_MODEL_REVISION}'
fi
test -x '${VENV}/bin/python'
test -f '${TRAIN_DATA}'
test -f '${VAL_DATA}'
test -f '${DATA_DIR}/eval_jsonl/math500.jsonl'
test -f '${DATA_DIR}/eval_jsonl/aime24.jsonl'
test -f '${DATA_DIR}/eval_jsonl/aime25.jsonl'
test -f '${DATA_DIR}/eval_jsonl/amc23.jsonl'
test \$(cat '${REMOTE_ROOT}/external/revisiting_opd.UPSTREAM_COMMIT') = '${SUBMODULE_BASE_COMMIT}'
if [[ -n '${EXPECTED_TRAIN_SHA256}' ]]; then
  actual_train_sha256=\$(sha256sum '${TRAIN_DATA}' | awk '{print \$1}')
  test \"\${actual_train_sha256}\" = '${EXPECTED_TRAIN_SHA256}'
fi
if git -C '${REMOTE_ROOT}' rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  bash '${REMOTE_ROOT}/scripts/setup_revisiting_opd.sh' >/dev/null
fi
mkdir -p '${RUN_DIR}' '${CKPTS_DIR}' '${LOG_DIR}' '${RUN_DIR}/summaries' \
  '${LOCAL_CACHE_ROOT}/tmp' '${LOCAL_CACHE_ROOT}/vllm_cache' \
  '${LOCAL_CACHE_ROOT}/torchinductor' '${LOCAL_CACHE_ROOT}/triton' \
  '${LOCAL_CACHE_ROOT}/cuda_cache' '${LOCAL_CACHE_ROOT}/outlines'
if [[ -f '${RUN_DIR}/train.pid' ]] && ps -p \"\$(cat '${RUN_DIR}/train.pid')\" >/dev/null 2>&1; then
  echo \"already_running pid=\$(cat '${RUN_DIR}/train.pid') run_dir=${RUN_DIR}\"
  exit 0
fi
mkdir -p '${RUN_DIR}/launch_history'
launch_archive=\$(date +%Y%m%dT%H%M%S)
for evidence in run_card.json command.sh env.txt artifact_hashes.sha256 script_hashes.sha256 \
  started_at.txt finished_at.txt exit_code.txt train.pid acceptance.json; do
  if [[ -f '${RUN_DIR}/'\"\${evidence}\" ]]; then
    cp '${RUN_DIR}/'\"\${evidence}\" '${RUN_DIR}/launch_history/'\"\${launch_archive}_\${evidence}\"
  fi
done
if [[ -f '${RUN_DIR}/logs/nohup.log' ]]; then
  cp '${RUN_DIR}/logs/nohup.log' '${RUN_DIR}/launch_history/'\"\${launch_archive}_nohup.log\"
fi
(cd '${REMOTE_ROOT}/external/revisiting_opd' && \
  sha256sum -c '${REMOTE_ROOT}/manifests/revisiting_opd_runtime.sha256') \
  > '${RUN_DIR}/revisiting_opd_manifest_check.txt'
cp '${DATA_DIR}/manifest.json' '${RUN_DIR}/data_manifest.json'
sha256sum \
  '${REMOTE_ROOT}/scripts/launch_revisiting_block_opd_formal_train.sh' \
  '${REMOTE_ROOT}/scripts/run_revisiting_sampled_block_opd_math.sh' \
  '${REMOTE_ROOT}/scripts/setup_revisiting_opd.sh' \
  '${REMOTE_ROOT}/scripts/launch_qwen3_math_eval.sh' \
  '${REMOTE_ROOT}/scripts/eval_qwen3_math_vllm.py' \
  '${REMOTE_ROOT}/scripts/audit_block10_run.py' \
  '${REMOTE_ROOT}/scripts/block3_replication_control.sh' \
  '${REMOTE_ROOT}/scripts/sliding_window_ml2_control.sh' \
  '${REMOTE_ROOT}/scripts/audit_window_run.py' \
  '${REMOTE_ROOT}/scripts/audit_window_control.py' \
  '${REMOTE_ROOT}/scripts/analyze_block10_collapse_diagnostics.py' \
  '${REMOTE_ROOT}/scripts/analyze_single_opd_diagnostics.py' \
  '${REMOTE_ROOT}/patches/revisiting_opd/blockwise_sampled_opd.patch' \
  '${REMOTE_ROOT}/external/revisiting_opd.UPSTREAM_COMMIT' \
  '${REMOTE_ROOT}/manifests/revisiting_opd_runtime.sha256' \
  '${REMOTE_ROOT}/opd_ext/diagnostics.py' \
  '${REMOTE_ROOT}/opd_ext/window_supervision.py' \
  '${REMOTE_ROOT}/opd_ext/math_protocol.py' \
  '${REMOTE_ROOT}/opd_ext/request_seeds.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/agent_system/environments/env_manager.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/agent_system/multi_turn_rollout/rollout_loop.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/trainer/ppo/core_algos.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/trainer/ppo/ray_trainer_multitask.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/workers/actor/dp_actor.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/workers/fsdp_workers.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/utils/reward_score/math.py' \
  '${REMOTE_ROOT}/external/revisiting_opd/verl/trainer/config/ppo_trainer.yaml' \
  > '${RUN_DIR}/script_hashes.sha256'
window_supervision_sha256=\$(sha256sum '${REMOTE_ROOT}/opd_ext/window_supervision.py' | awk '{print \$1}')
{
  sha256sum '${TRAIN_DATA}' '${VAL_DATA}' \
    '${DATA_DIR}/eval_jsonl/math500.jsonl' \
    '${DATA_DIR}/eval_jsonl/aime24.jsonl' \
    '${DATA_DIR}/eval_jsonl/aime25.jsonl' \
    '${DATA_DIR}/eval_jsonl/amc23.jsonl'
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
  \"opd_block_size\": ${OPD_BLOCK_SIZE},
  \"opd_block_advantage_mode\": \"${OPD_BLOCK_ADVANTAGE_MODE}\",
  \"opd_window_mode\": \"${OPD_WINDOW_MODE}\",
  \"opd_window_seed\": ${OPD_WINDOW_SEED},
  \"ppo_epochs\": 1,
  \"window_supervision_sha256\": \"\${window_supervision_sha256}\",
  \"source_commit\": \"${SOURCE_COMMIT}\",
  \"revisiting_opd_base_commit\": \"${SUBMODULE_BASE_COMMIT}\",
  \"project_name\": \"${PROJECT_NAME}\",
  \"experiment_name\": \"${EXP_NAME}\",
  \"student_model\": \"${STUDENT_MODEL}\",
  \"teacher_model\": \"${MATH_TEACHER}\",
  \"student_model_revision\": \"${STUDENT_MODEL_REVISION}\",
  \"teacher_model_revision\": \"${TEACHER_MODEL_REVISION}\",
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
  \"rollout_temperature\": ${ROLLOUT_TEMPERATURE},
  \"rollout_top_p\": ${ROLLOUT_TOP_P},
  \"actor_ppo_micro_batch_size_per_gpu\": ${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU},
  \"rollout_log_prob_micro_batch_size_per_gpu\": ${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU},
  \"ref_log_prob_micro_batch_size_per_gpu\": ${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU},
  \"opd_diagnostics\": ${OPD_DIAGNOSTICS},
  \"opd_diag_interval\": ${OPD_DIAG_INTERVAL},
  \"opd_diag_topk\": ${OPD_DIAG_TOPK},
  \"opd_diag_position_bin\": ${OPD_DIAG_POSITION_BIN},
  \"opd_diag_position_stride\": ${OPD_DIAG_POSITION_STRIDE},
  \"opd_diag_sign_epsilon\": ${OPD_DIAG_SIGN_EPS},
  \"diagnostic_save_steps\": \"${DIAGNOSTIC_SAVE_STEPS}\",
  \"stop_after_step\": ${STOP_AFTER_STEP},
  \"filter_overlong_prompts\": ${FILTER_OVERLONG_PROMPTS},
  \"expected_train_sha256\": \"${EXPECTED_TRAIN_SHA256}\",
  \"resume_mode\": \"${RESUME_MODE}\",
  \"resume_from_path\": \"${RESUME_FROM_PATH}\",
  \"diagnostic_output_dir\": \"${OPD_DIAG_OUTPUT_DIR}\",
  \"opd_prompt_protocol\": \"${OPD_PROMPT_PROTOCOL}\",
  \"request_seed_rule\": \"${OPD_REQUEST_SEED_RULE}\",
  \"lossless_rollout_dir\": \"${LOSSLESS_ROLLOUT_DIR}\",
  \"rollout_attempt_id\": \"${ROLLOUT_ATTEMPT_ID}\",
  \"checkpoint_policy\": \"preserve all milestone checkpoints; no automatic deletion\",
  \"baseline_alignment\": \"${BASELINE_ALIGNMENT}\"
}
JSON
cat > '${RUN_DIR}/command.sh' <<'CMD'
#!/usr/bin/env bash
set -euo pipefail
RUN_DIR='${RUN_DIR}'
rm -f "\${RUN_DIR}/exit_code.txt" "\${RUN_DIR}/finished_at.txt"
date --iso-8601=seconds > "\${RUN_DIR}/started_at.txt"
record_exit() {
  local status=\$?
  trap - EXIT
  printf '%s\n' "\${status}" > "\${RUN_DIR}/exit_code.txt"
  date --iso-8601=seconds > "\${RUN_DIR}/finished_at.txt"
  exit "\${status}"
}
trap record_exit EXIT
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
OPD_PROMPT_PROTOCOL='${OPD_PROMPT_PROTOCOL}' \
OPD_REQUEST_SEED_RULE='${OPD_REQUEST_SEED_RULE}' \
LOSSLESS_ROLLOUT_DIR='${LOSSLESS_ROLLOUT_DIR}' \
ROLLOUT_ATTEMPT_ID='${ROLLOUT_ATTEMPT_ID}' \
SOURCE_COMMIT='${SOURCE_COMMIT}' \
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
OPD_WINDOW_SEED='${OPD_WINDOW_SEED}' \
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
ROLLOUT_TEMPERATURE='${ROLLOUT_TEMPERATURE}' \
ROLLOUT_TOP_P='${ROLLOUT_TOP_P}' \
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU='${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}' \
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU='${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}' \
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU='${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}' \
TRAINER_LOGGER=\"${TRAINER_LOGGER}\" \
OPD_DIAGNOSTICS='${OPD_DIAGNOSTICS}' \
OPD_DIAG_INTERVAL='${OPD_DIAG_INTERVAL}' \
OPD_DIAG_TOPK='${OPD_DIAG_TOPK}' \
OPD_DIAG_POSITION_BIN='${OPD_DIAG_POSITION_BIN}' \
OPD_DIAG_POSITION_STRIDE='${OPD_DIAG_POSITION_STRIDE}' \
OPD_DIAG_SIGN_EPS='${OPD_DIAG_SIGN_EPS}' \
DIAGNOSTIC_SAVE_STEPS='${DIAGNOSTIC_SAVE_STEPS}' \
STOP_AFTER_STEP='${STOP_AFTER_STEP}' \
FILTER_OVERLONG_PROMPTS='${FILTER_OVERLONG_PROMPTS}' \
RESUME_MODE='${RESUME_MODE}' \
RESUME_FROM_PATH='${RESUME_FROM_PATH}' \
OPD_DIAG_OUTPUT_DIR='${OPD_DIAG_OUTPUT_DIR}' \
bash scripts/run_revisiting_sampled_block_opd_math.sh
CMD
'${VENV}/bin/python' -m json.tool '${RUN_DIR}/run_card.json' >/dev/null
bash -n '${RUN_DIR}/command.sh'
chmod +x '${RUN_DIR}/command.sh'
if [[ '${PREPARE_ONLY}' == true ]]; then
  echo 'prepared command=${RUN_DIR}/command.sh run_dir=${RUN_DIR}'
  exit 0
fi
nohup bash '${RUN_DIR}/command.sh' > '${LOG_DIR}/nohup.log' 2>&1 &
echo \$! > '${RUN_DIR}/train.pid'
echo \"pid=\$(cat '${RUN_DIR}/train.pid') run_dir=${RUN_DIR} log=${LOG_DIR}/nohup.log\"
"
