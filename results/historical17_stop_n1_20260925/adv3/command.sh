#!/usr/bin/env bash
set -euo pipefail
RUN_DIR='/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260925v1_historical17_components_n1_batch32_seed21_ml2/adv3'
rm -f ${RUN_DIR}/exit_code.txt ${RUN_DIR}/finished_at.txt
date --iso-8601=seconds > ${RUN_DIR}/started_at.txt
record_exit() {
  local status=$?
  trap - EXIT
  printf '%s\n' ${status} > ${RUN_DIR}/exit_code.txt
  date --iso-8601=seconds > ${RUN_DIR}/finished_at.txt
  exit ${status}
}
trap record_exit EXIT
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PATH='/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl/bin':$PATH
export PYTHONPATH='/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/deployments/7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5/external/revisiting_opd':${PYTHONPATH:-}
export HF_HOME='/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/hf_home'
export HF_HUB_CACHE='/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/hf_home/hub'
export HF_DATASETS_CACHE='/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/hf_home/datasets'
export TMPDIR='/dev/shm/ha/adv3/train/tmp'
export VLLM_CACHE_ROOT='/dev/shm/ha/adv3/train/vllm_cache'
export TORCHINDUCTOR_CACHE_DIR='/dev/shm/ha/adv3/train/torchinductor'
export TRITON_CACHE_DIR='/dev/shm/ha/adv3/train/triton'
export CUDA_CACHE_PATH='/dev/shm/ha/adv3/train/cuda_cache'
export OUTLINES_CACHE_DIR='/dev/shm/ha/adv3/train/outlines'
export PYTHONUNBUFFERED=1
cd '/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/deployments/7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5'
VARIANT='adv3' OPD_PROMPT_PROTOCOL='qwen3_historical17_v1' OPD_REQUEST_SEED_RULE='legacy' LOSSLESS_ROLLOUT_DIR='/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260925v1_historical17_components_n1_batch32_seed21_ml2/adv3/rollouts' ROLLOUT_ATTEMPT_ID='formal' SOURCE_COMMIT='7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5' PROJECT_NAME='opd_historical_components' EXP_NAME='historical17-adv3' STUDENT_MODEL='/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/models/Qwen3-1.7B-Base' MATH_TEACHER='/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO' TRAIN_DATA='/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/data/math_opd_dapo17k_hf_full_eval4/train.parquet' VAL_DATA='/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/data/math_opd_dapo17k_hf_full_eval4/test.parquet' CKPTS_DIR='/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260925v1_historical17_components_n1_batch32_seed21_ml2/adv3/checkpoints' LOG_DIR='/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260925v1_historical17_components_n1_batch32_seed21_ml2/adv3/logs' N_GPUS_PER_NODE='4' RAY_NUM_CPUS='64' ENV_SEED='21' OPD_WINDOW_SEED='910021' TRAIN_BATCH_SIZE='32' PPO_MINI_BATCH_SIZE='32' ROLLOUT_GROUP_SIZE='1' MAX_PROMPT_LENGTH='2048' MAX_RESPONSE_LENGTH='16384' LEARNING_RATE='2e-6' TOTAL_TRAINING_STEPS='200' SAVE_FREQ='-1' TEST_FREQ='-1' VAL_N='1' ROLLOUT_GPU_MEMORY_UTILIZATION='0.6' ROLLOUT_MAX_NUM_BATCHED_TOKENS='18432' ROLLOUT_TEMPERATURE='1.0' ROLLOUT_TOP_P='0.9' ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU='1' ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU='4' REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU='1' TRAINER_LOGGER="['console']" OPD_DIAGNOSTICS='true' OPD_DIAG_INTERVAL='5' OPD_DIAG_TOPK='16' OPD_DIAG_POSITION_BIN='128' OPD_DIAG_POSITION_STRIDE='1' OPD_DIAG_SIGN_EPS='1e-4' DIAGNOSTIC_SAVE_STEPS='50,100,150,200' STOP_AFTER_STEP='-1' FILTER_OVERLONG_PROMPTS='false' RESUME_MODE='disable' RESUME_FROM_PATH='' OPD_DIAG_OUTPUT_DIR='/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260925v1_historical17_components_n1_batch32_seed21_ml2/adv3/diagnostics' bash scripts/run_revisiting_sampled_block_opd_math.sh
