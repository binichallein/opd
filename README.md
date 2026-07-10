# OPD 内部实验仓库

这是一个内部研究仓库，用来同步、规范和复现 On-Policy Distillation
（OPD）相关实验，重点是语言模型后训练中的 token-level / block-level
reverse-KL 目标。

当前仓库包含 clean-room OPD 训练与评测脚本、block supervision 单测、
整理后的结果摘要，以及 block reverse-KL 实验的 HTML 报告。
同时，`external/revisiting_opd` 固定了公开 `revisiting_opd` codebase，
用于后续论文级 baseline 与 DAPO-Math-17K 对齐实验。

## 当前论文级实验设置

为了和 `Rethinking OPD` / `Blockwise Policy-Drift Gating` 的数学 OPD
实验线对齐，当前 DAPO-Math-17K 对比使用：

- Student：`Qwen3-1.7B-Base`
- Teacher：`Qwen3-4B-Base-GRPO`
- 训练数据：DAPO-Math-17K released parquet 展开的 1,791,700-row
  prompt-answer pool
  `/mnt/data/cpfs/Yaleon/opd/data/math_opd_dapo17k_hf_full_eval4/train.parquet`
- Eval 数据：`/mnt/data/cpfs/Yaleon/opd/data/math_opd_dapo17k_hf_full_eval4/test.parquet`
  含 MATH500 500、AIME24 30、AIME25 30、AMC23 83
- 正式比较四组：标准 `token_opd`、`block3_mean`、`block5_mean`、
  `block10_mean`
- `external/revisiting_opd/` 只作为代码底座；不采用该论文的
  Qwen2.5/OpenThinker 模型设置作为本轮主实验模型

旧的官方 `Qwen/Qwen3-0.6B` + `Qwen/Qwen3-4B` 结果属于 clean-room pilot，
不能作为论文同款模型结果引用。

注：`math_opd_dapo17k_hf_dedup_eval4` 是早期去重 pilot 目录，不再作为
当前论文级主设置。和 `Blockwise Policy-Drift Gating` 对齐时，默认使用
raw row-pool 训练表。

四组均已训练到 step 200，并在 MATH500、AIME24、AIME25、AMC23 上完成
`n=8` full eval。当前单 seed 结果为：`block3_mean` 相对 token OPD 有小幅
macro 增益，`block5_mean` 回退，`block10_mean` 在约 step 50-60 后发生
高熵、长度膨胀和全格式错误的生成崩塌。这个结果不支持“block 越大越好”；
严格归因仍需在同一硬件上增加多个 seed。完整配置、结果和诊断见
`reports/block_opd_experiment_report.html`。

## 仓库里保存什么

- `scripts/`：训练、评测、regrade、分析和远端 launch 脚本
- `tests/`：关键逻辑的单元测试，当前重点是 block supervision
- `manuscript/`：研究笔记和论文草稿
- `results/*.json`：小型、整理后的结果摘要表
- `reports/`：方便阅读和讨论的 HTML 报告
- `figures/`：论文或汇报用 SVG 图
- `docs/`：实验协议、数据策略、模型策略、结果记录
- `configs/`：可复现实验配置
- `external/revisiting_opd`：公开 `revisiting_opd` submodule
- `patches/revisiting_opd/`：我们对外部 baseline codebase 的最小补丁
- `AGENTS.md`：给 AI coding agent 看的项目规则

## 仓库里不保存什么

不要提交以下内容：

- 原始数据或处理后数据
- checkpoint
- 模型权重
- 完整 prediction `.jsonl`
- 训练或评测日志
- cache 目录
- SSH key、access key、API token、`.env`

`.gitignore` 会默认拦截这些文件。这个仓库保存的是实验逻辑、配置、
结论和小型摘要，不用来复制大体积实验产物。

## 当前主要结果

最新完成的实验验证了 block-level reverse-KL OPD 中的 advantage 聚合方式：

- `naive block-3`：把 3 个连续 token 的 advantage 直接相加
- `mean block-3`：把 3 个连续 token 的 advantage 取平均
- `mixed block-3`：把 token 自己的 advantage 和 block 平均 advantage 混合

2026-07-08 的完整实验结果显示：

- `mean block-3` 相比 naive block-3 明显降低 gradient norm；
- 同时在 GSM8K 和 MATH500 上都略有提升；
- `mixed block-3, lambda=0.5` 梯度最低，但 MATH500 下降，需要继续 sweep。

因此当前结论不是“block 越大越好”，而是：

```text
block-level reverse-KL OPD 需要 variance-controlled advantage aggregation。
```

详见：

- `reports/block_opd_experiment_report.html`
- `docs/results/2026-07-08-block-advantage.md`

## 快速检查

```bash
python -m py_compile scripts/clean_opd_train.py scripts/summarize_block_advantage_experiment.py
python -m pytest tests/test_block_supervision.py -q
```

当前实验机上使用的是 `vllm` conda 环境。

## revisiting_opd 基线代码

准备外部 baseline 代码：

```bash
bash scripts/setup_revisiting_opd.sh
```

这会初始化 `external/revisiting_opd` submodule，并应用
`patches/revisiting_opd/blockwise_sampled_opd.patch`。补丁默认不改变上游行为：
`actor_rollout_ref.actor.opd_block_size=1` 时仍是标准 sampled-token OPD。

开启我们的 block-level sampled OPD：

```text
algorithm.adv_estimator=opd
actor_rollout_ref.actor.use_kl_loss=False
algorithm.use_kl_in_reward=True
actor_rollout_ref.actor.opd_block_size=3
actor_rollout_ref.actor.opd_block_advantage_mode=mean
```

推荐入口是：

```bash
VARIANT=block3_mean bash scripts/run_revisiting_sampled_block_opd_math.sh
```

不要把 `opd_block_size>1` 直接加到 `placeholder + full_reverse KL` 路径上；
那条路径是 token-level full/top-k KL baseline，不是 sampled block reverse-KL。

推荐论文级比较矩阵见
`configs/experiments/revisiting_opd/blockwise_sampled_opd.yaml`。

## 远端训练规范

长时间训练必须在训练服务器上 detached 启动，例如使用 `nohup` 或调度系统；
不能依赖本地 SSH 会话存活。

每个真实训练都必须：

- 保存完整 checkpoint state，而不是只保存 HF 权重；
- 记录命令、配置、数据 manifest、模型来源、seed 和 run root；
- 在大规模训练前先做小型 resume gate；
- clean-room pilot 的完整 eval 默认指 GSM8K 1319 条和 MATH500 500 条；
  DAPO-Math-17K 论文级实验的完整 eval 指 MATH500、AIME24、AIME25、AMC23。

详见 `docs/experiment_protocol.md`。
