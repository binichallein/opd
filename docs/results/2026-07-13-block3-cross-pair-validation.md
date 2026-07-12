# 2026-07-13 Block3 跨师生配对验证

## 问题

验证 `block3_mean` 是否能在不同 teacher-student pair 上稳定优于标准
sampled-token OPD。Block3 将连续 3 个 sampled token 的
teacher-vs-old-student advantage 取平均，并对 3-token joint policy ratio
应用同一个 block 级更新信号。

## 固定实验合同

- 数据：DAPO-Math-17K raw 1,791,700-row pool。
- 训练：seed 21，200 steps，每步 4 prompt，每个 prompt 8 rollout，
  LR `2e-6`，最大响应 16,384。
- Eval：Math500 500、AIME24 30、AIME25 30、AMC23 83；每题 n=8，
  rollout seed 21-28；Step 50/100/200。
- 评分：固定历史 grader；10,000 次 task-stratified paired prompt
  bootstrap。
- 配对审计：每对 token/Block3 的 41 个训练 prompt batch hash 全部匹配。

## Step 200 主结果

| Pair | Avg@8 delta | 95% CI | Pass@8 delta | 95% CI | 判定 |
|---|---:|---:|---:|---:|---|
| Qwen3-1.7B ← Qwen3-4B-GRPO | +0.0514 | [+0.0353, +0.0680] | +0.0584 | [+0.0141, +0.1039] | 强单 seed 支持 |
| DeepSeek-R1-Distill-Qwen-1.5B ← JustRL-DeepSeek-1.5B | +0.0158 | [-0.0031, +0.0356] | -0.0015 | [-0.0345, +0.0323] | 未复现 |

DeepSeek/JustRL 的 Step 50 Pass@8 显著下降，Step 100 Avg@8 也显著下降。
Step 200 虽然 Avg@8 点估计转正，但 CI 跨 0，Pass@8 点估计仍为负。
因此它不满足预注册的方向复现门槛。

## 机制与成本

DeepSeek/JustRL 的 Block3 在 Step 200 仍有 25.5% sign flip、11.3%
weighted sign flip 和 1.035 normalized leakage；token baseline 对应指标为 0。
两条训练都没有 NaN/Inf、ratio overflow 或 entropy collapse，未复现不是数值
崩塌导致的。

完整 200 步耗时为 token 18,951 秒、Block3 18,767 秒，差约 -0.97%。
当前实现仍对完整学生 rollout 做 teacher forward，再在 loss 中聚合连续 token；
它没有减少 teacher forward 次数，也没有带来实质训练加速。

## 结论

`block3_mean` 在 Qwen3 师生对上有可靠单 seed 增益，但未在
DeepSeek/JustRL 上复现。当前最准确的表述是：

```text
Block3 mean 是模型对相关的 credit-assignment 改动，尚不是跨师生稳健的
通用 OPD 改进。
```

下一步研究应围绕 sign agreement / leakage-aware gating，而不是继续假设固定
block size 会普遍优于 token OPD。还需要至少 3 个 paired training seeds 来估计
训练方差。

详细报告：

- `reports/block_opd_experiment_report.html`
- `reports/paired_validation/ml2/report.html`
- `reports/paired_validation/deepseek_justrl/report.html`

每个配对目录还保留 `report.audited.html`，它是远端最终化报告的逐字节
副本，可由 `finalization_hashes.sha256` 校验；默认的 `report.html` 使用相同
输入重新生成，只修复了窄屏布局。
