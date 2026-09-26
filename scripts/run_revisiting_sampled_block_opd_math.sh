#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISITING_DIR="${ROOT_DIR}/external/revisiting_opd"
export PYTHONPATH="${ROOT_DIR}:${REVISITING_DIR}:${PYTHONPATH:-}"

if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
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
actor_ppo_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
rollout_log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
ref_log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
opd_diagnostics="${OPD_DIAGNOSTICS:-false}"
opd_diag_interval="${OPD_DIAG_INTERVAL:-5}"
opd_diag_topk="${OPD_DIAG_TOPK:-16}"
opd_diag_position_bin="${OPD_DIAG_POSITION_BIN:-128}"
opd_diag_position_stride="${OPD_DIAG_POSITION_STRIDE:-1}"
opd_diag_sign_eps="${OPD_DIAG_SIGN_EPS:-1e-4}"
diagnostic_save_steps="${DIAGNOSTIC_SAVE_STEPS:-40,50,60,80,100,200}"
diagnostic_save_steps_list="[${diagnostic_save_steps}]"
opd_diag_output_dir="${OPD_DIAG_OUTPUT_DIR:-${CKPTS_DIR}/../diagnostics}"
stop_after_step="${STOP_AFTER_STEP:--1}"
filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-true}"
resume_mode="${RESUME_MODE:-disable}"
resume_from_path="${RESUME_FROM_PATH:-}"
window_mode=fixed
block_ablation=legacy
window_seed="${OPD_WINDOW_SEED:-$((910000 + ${ENV_SEED:-21}))}"
protocol_args=()
if [[ "${OPD_PROMPT_PROTOCOL:-legacy}" == math_eval_nonthinking_v1 || "${OPD_PROMPT_PROTOCOL:-legacy}" == llama32_nonthinking_v1 || "${OPD_PROMPT_PROTOCOL:-legacy}" == llama32_historical17_v1 || "${OPD_PROMPT_PROTOCOL:-legacy}" == qwen3_historical17_v1 || "${OPD_PROMPT_PROTOCOL:-legacy}" == qwen3_completion_boxed_v1 || "${OPD_PROMPT_PROTOCOL:-legacy}" == qwen3_native_chat_no_thinking_boxed_v1 ]]; then
    : "${LOSSLESS_ROLLOUT_DIR:?Complete rollout retention is required}"
    : "${ROLLOUT_ATTEMPT_ID:?An explicit attempt identity is required}"
    : "${SOURCE_COMMIT:?Source identity is required}"
    stop_ids='[151643,151645]'
    if [[ "${OPD_PROMPT_PROTOCOL}" == llama32_nonthinking_v1 || "${OPD_PROMPT_PROTOCOL}" == llama32_historical17_v1 ]]; then
        stop_ids='[128001,128008,128009]'
    elif [[ "${OPD_PROMPT_PROTOCOL}" == qwen3_historical17_v1 || "${OPD_PROMPT_PROTOCOL}" == qwen3_completion_boxed_v1 ]]; then
        stop_ids='[]'
    elif [[ "${OPD_PROMPT_PROTOCOL}" == qwen3_native_chat_no_thinking_boxed_v1 ]]; then
        stop_ids='[151645,151643]'
    fi
    protocol_args=(
        "+data.opd_prompt_protocol=${OPD_PROMPT_PROTOCOL}"
        +actor_rollout_ref.rollout.retain_generation_metadata=true
        "+actor_rollout_ref.rollout.stop_token_ids=${stop_ids}"
        "+trainer.lossless_rollout_dir=${LOSSLESS_ROLLOUT_DIR}"
        "+trainer.rollout_attempt_id=${ROLLOUT_ATTEMPT_ID}"
        "+trainer.source_commit=${SOURCE_COMMIT}"
    )
    if [[ "${OPD_PROMPT_PROTOCOL}" == qwen3_historical17_v1 || "${OPD_PROMPT_PROTOCOL}" == qwen3_completion_boxed_v1 ]]; then
        protocol_args+=(+actor_rollout_ref.rollout.preserve_legacy_response_mask=true)
        if [[ "${OPD_PROMPT_PROTOCOL}" == qwen3_completion_boxed_v1 ]]; then
            protocol_args+=(+data.apply_chat_template_kwargs.enable_thinking=false)
        fi
    elif [[ "${OPD_PROMPT_PROTOCOL}" != llama32_historical17_v1 ]]; then
        protocol_args+=(+data.apply_chat_template_kwargs.enable_thinking=false)
    fi
    if [[ "${OPD_PROMPT_PROTOCOL}" == qwen3_native_chat_no_thinking_boxed_v1 ]]; then
        protocol_args+=(
            "+actor_rollout_ref.rollout.opd_prompt_protocol=${OPD_PROMPT_PROTOCOL}"
            +actor_rollout_ref.rollout.preserve_legacy_response_mask=false
        )
    fi
elif [[ "${OPD_PROMPT_PROTOCOL:-legacy}" != legacy ]]; then
    echo 'Unknown prompt protocol' >&2
    exit 2
fi

if [[ "${OPD_REQUEST_SEED_RULE:-legacy}" != legacy ]]; then
    if [[ "${OPD_REQUEST_SEED_RULE}" != sha256_step_question_sample_v1 ||
          ( "${OPD_PROMPT_PROTOCOL:-legacy}" != qwen3_completion_boxed_v1 &&
            "${OPD_PROMPT_PROTOCOL:-legacy}" != qwen3_native_chat_no_thinking_boxed_v1 ) ]]; then
        echo 'Unapproved request seed rule/protocol combination' >&2
        exit 2
    fi
    protocol_args+=("+actor_rollout_ref.rollout.request_seed_rule=${OPD_REQUEST_SEED_RULE}")
fi

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
  adv3)
    block_size=3
    block_mode=mean
    block_mix_lambda=0.5
    block_ablation=adv_only
    ;;
  joint3)
    block_size=3
    block_mode=mean
    block_mix_lambda=0.5
    block_ablation=joint_tokenmean
    ;;
  scale3)
    block_size=3
    block_mode=mean
    block_mix_lambda=0.5
    block_ablation=token_scale
    ;;
  random3|sliding3)
    block_size=3
    block_mode=mean
    block_mix_lambda=0.5
    window_mode="${VARIANT%3}"
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
    "${protocol_args[@]}" \
    algorithm.adv_estimator=opd \
    actor_rollout_ref.actor.kl_loss_type=k1 \
    +actor_rollout_ref.actor.kl_topk_tokens=32 \
    +actor_rollout_ref.actor.norm_to_one_for_kl=True \
    +actor_rollout_ref.actor.clip_log_ratio=False \
    +actor_rollout_ref.actor.opd_mask_special_tokens=False \
    actor_rollout_ref.rollout.temperature="${rollout_temperature}" \
    actor_rollout_ref.rollout.top_p="${rollout_top_p}" \
    +actor_rollout_ref.rollout.seed="${ENV_SEED:-21}" \
    actor_rollout_ref.ref.model.path="${MATH_TEACHER}" \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_data_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts="${filter_overlong_prompts}" \
    data.truncation=middle \
    data.return_raw_chat=True \
    +data.seed="${ENV_SEED:-21}" \
    +data.batching_mode=sequential \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.actor.optim.lr="${learning_rate}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${actor_ppo_micro_batch_size_per_gpu}" \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=1 \
    actor_rollout_ref.actor.opd_block_size="${block_size}" \
    actor_rollout_ref.actor.opd_block_advantage_mode="${block_mode}" \
    actor_rollout_ref.actor.opd_block_mix_lambda="${block_mix_lambda}" \
    +actor_rollout_ref.actor.opd_block_ablation="${block_ablation}" \
    actor_rollout_ref.actor.opd_window_mode="${window_mode}" \
    actor_rollout_ref.actor.opd_window_seed="${window_seed}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${rollout_log_prob_micro_batch_size_per_gpu}" \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ref_log_prob_micro_batch_size_per_gpu}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    algorithm.use_kl_in_reward=True \
    +algorithm.opd_diagnostics.enabled="${opd_diagnostics}" \
    +algorithm.opd_diagnostics.interval="${opd_diag_interval}" \
    +algorithm.opd_diagnostics.topk="${opd_diag_topk}" \
    +algorithm.opd_diagnostics.position_bin="${opd_diag_position_bin}" \
    +algorithm.opd_diagnostics.position_stride="${opd_diag_position_stride}" \
    +algorithm.opd_diagnostics.sign_epsilon="${opd_diag_sign_eps}" \
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
    +trainer.checkpoint_milestones="${diagnostic_save_steps_list}" \
    trainer.test_freq="${TEST_FREQ:-1000000}" \
    trainer.total_training_steps="${total_training_steps}" \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.default_local_dir="${CKPTS_DIR}" \
    +trainer.opd_diagnostic_dir="${opd_diag_output_dir}" \
    +trainer.stop_after_step="${stop_after_step}" \
    trainer.resume_mode="${resume_mode}" \
    trainer.resume_from_path="${resume_from_path}" \
    +trainer.val_generation_dir="${VAL_GENERATION_DIR:-${CKPTS_DIR}/val_generations}" \
    ray_init.num_cpus="${RAY_NUM_CPUS:-96}" \
    +ray_init.include_dashboard=False \
    2>&1 | tee "${LOG_DIR}/${exp_name}_${TIME_STAMP}.log"
