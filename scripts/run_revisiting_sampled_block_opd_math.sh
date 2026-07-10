#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISITING_DIR="${ROOT_DIR}/external/revisiting_opd"

if [[ -d "${ROOT_DIR}/.git" ]]; then
  bash "${ROOT_DIR}/scripts/setup_revisiting_opd.sh" >/dev/null
fi
cd "${REVISITING_DIR}"

ENGINE="${ENGINE:-vllm}"
VARIANT="${VARIANT:-block3_mean}"
TIME_STAMP="$(date +"%m%d_%H%M%S")"

project_name="${PROJECT_NAME:-opd_block_sampled}"
exp_name="${EXP_NAME:-sampled-${VARIANT}-math}"

STUDENT_MODEL="${STUDENT_MODEL:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-1.7B-Base}"
MATH_TEACHER="${MATH_TEACHER:-/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/models/Qwen3-4B-Base-GRPO}"
TRAIN_DATA="${TRAIN_DATA:-/path/to/math_opd/train.parquet}"
VAL_DATA="${VAL_DATA:-/path/to/math_opd/test.parquet}"
CKPTS_DIR="${CKPTS_DIR:-${PWD}/ckpts/${exp_name}_${TIME_STAMP}}"
LOG_DIR="${LOG_DIR:-${PWD}/logs/math}"

train_data_size="${TRAIN_BATCH_SIZE:-4}"
val_data_size="${VAL_BATCH_SIZE:-128}"
group_size="${ROLLOUT_GROUP_SIZE:-8}"
num_cpus_per_env_worker="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
rollout_top_p="${ROLLOUT_TOP_P:-0.9}"
rollout_temperature="${ROLLOUT_TEMPERATURE:-1.0}"
val_top_p="${VAL_TOP_P:-0.9}"
val_temperature="${VAL_TEMPERATURE:-1.0}"
val_n="${VAL_N:-8}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-16384}"
learning_rate="${LEARNING_RATE:-2e-6}"
total_training_steps="${TOTAL_TRAINING_STEPS:-200}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
rollout_max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}"

case "${VARIANT}" in
  token_opd)
    block_size=1
    block_mode=sum
    block_mix_lambda=0.5
    ;;
  block3_sum)
    block_size=3
    block_mode=sum
    block_mix_lambda=0.5
    ;;
  block3_mean)
    block_size=3
    block_mode=mean
    block_mix_lambda=0.5
    ;;
  block5_mean)
    block_size=5
    block_mode=mean
    block_mix_lambda=0.5
    ;;
  block10_mean)
    block_size=10
    block_mode=mean
    block_mix_lambda=0.5
    ;;
  block3_mixed_lam05)
    block_size=3
    block_mode=mixed
    block_mix_lambda=0.5
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}"

set -x
python3 -m verl.trainer.main_ppo_multitask \
    algorithm.adv_estimator=opd \
    actor_rollout_ref.actor.kl_loss_type=k1 \
    +actor_rollout_ref.actor.kl_topk_tokens=32 \
    +actor_rollout_ref.actor.norm_to_one_for_kl=True \
    +actor_rollout_ref.actor.clip_log_ratio=False \
    +actor_rollout_ref.actor.opd_mask_special_tokens=False \
    actor_rollout_ref.rollout.temperature="${rollout_temperature}" \
    actor_rollout_ref.rollout.top_p="${rollout_top_p}" \
    actor_rollout_ref.ref.model.path="${MATH_TEACHER}" \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_data_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.truncation=middle \
    data.return_raw_chat=True \
    +data.batching_mode=sequential \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.actor.optim.lr="${learning_rate}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=1 \
    actor_rollout_ref.actor.opd_block_size="${block_size}" \
    actor_rollout_ref.actor.opd_block_advantage_mode="${block_mode}" \
    actor_rollout_ref.actor.opd_block_mix_lambda="${block_mix_lambda}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_memory_utilization}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens="${rollout_max_num_batched_tokens}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature="${val_temperature}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
    actor_rollout_ref.rollout.val_kwargs.n="${val_n}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    algorithm.use_kl_in_reward=True \
    env.env_name=math \
    env.seed="${ENV_SEED:-21}" \
    env.max_steps=30 \
    env.rollout.n="${group_size}" \
    env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
    trainer.critic_warmup=0 \
    trainer.logger="${TRAINER_LOGGER:-['console']}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-100}" \
    trainer.test_freq="${TEST_FREQ:-1000000}" \
    trainer.total_training_steps="${total_training_steps}" \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.val_generation_dir="${VAL_GENERATION_DIR:-${CKPTS_DIR}/val_generations}" \
    ray_init.num_cpus="${RAY_NUM_CPUS:-96}" \
    2>&1 | tee "${LOG_DIR}/${exp_name}_${TIME_STAMP}.log"
