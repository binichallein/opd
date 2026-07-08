# revisiting_opd 接入方案

我们把 `https://github.com/hhh675597/revisiting_opd` 固定为
`external/revisiting_opd` submodule，并把我们的改动保存为
`patches/revisiting_opd/blockwise_sampled_opd.patch`。

这样做的原因：

- 上游代码保持可追踪，便于和论文 baseline 对齐。
- 我们只提交小 patch、配置和报告，不把外部大仓改成不可读的混合状态。
- train 上可以通过 `scripts/setup_revisiting_opd.sh` 复现同一份代码状态。
- 原始训练数据、rollout、checkpoint 和预测 jsonl 继续不进 Git。

补丁内容：

- `compute_policy_loss` 增加 `opd_block_size`、`opd_block_advantage_mode`、`opd_block_mix_lambda`。
- `block_size=1` 时完全回到上游 sampled-token OPD。
- `block_size>1` 时，连续 token 先聚合成 block：
  - `old_block_logprob = sum old_log_prob[t:t+k]`
  - `current_block_logprob = sum log_prob[t:t+k]`
  - `block_advantage = sum/mean/mixed advantage[t:t+k]`
  - policy ratio 用 `exp(current_block_logprob - old_block_logprob)`。
- FSDP actor 和 Megatron actor 都透传这些配置。
- partial block 使用 `有效 token 数 / block_size` 作为 loss 权重，避免最后 1 个 token
  和完整 3-token block 等权。
- full/top-k KL 路径不是 sampled block reverse-KL；如果 `block_size>1` 与 full/top-k KL
  同时开启，代码会显式报错。

下一步实验矩阵：

- `token_opd`: 标准 sampled-token OPD，`opd_block_size=1`。
- `block3_sum`: naive block estimator，预期梯度更大。
- `block3_mean`: mean-normalized block estimator，是当前最值得验证的版本。
- `Teacher-TopK / LSM`: 来自 revisiting_opd 的外部强 baseline。

有效运行入口：

- 使用 `scripts/run_revisiting_sampled_block_opd_math.sh`。
- 必须保持 `algorithm.adv_estimator=opd`、`actor_rollout_ref.actor.use_kl_loss=False`、
  `algorithm.use_kl_in_reward=True`。
- 不要用 `examples/opd/opd_math_qwen2.5-7b_it.sh` 加 block 参数；那个脚本走
  `placeholder + full_reverse KL` 路径，只适合作为 token-level full/top-k KL baseline。

论文级数据建议使用 DAPO-Math-17K，并评估 MATH500、AIME24、AIME25、AMC23。
