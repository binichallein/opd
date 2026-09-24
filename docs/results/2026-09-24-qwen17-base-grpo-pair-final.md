# 4B-GRPO到1.7B-Base完整对照结果

## 范围与验收

北京时间2026-09-24 20:40:00，v5队列完成。教师为公共
`lllyx/Qwen3-4B-Base-GRPO`，学生为官方`Qwen/Qwen3-1.7B-Base`。
教师筛选的续写增益仍为inconclusive，用户明确授权继续训练；训练结果不回写筛选门槛。

- 两组分别从原始学生初始化，各200步、seed21、相同DAPO文件及实际源题目顺序。
- Base使用`qwen3_completion_boxed_v1`：裸题目、boxed指令、`Solution:`，无ChatML/think。
- 保留历史Block3 Mean：mean advantage、联合block PPO ratio及原损失归一化，
  不能将本次比较解释成仅对advantage进行平滑。
- 每组50/100/150/200的四卡完整训练状态均通过验收，不删除中间权重。
- 两组各6400条训练rollout，配对输入与请求seed的200步核验通过；每步标量与位置诊断均保留。
- 九次评测均使用历史external grader、n=8、temperature=1、top-p=0.9、max_tokens=16384、
  seeds21至28。每次643题、5144条rollout，总计46296条评测rollout，均验收通过。
- MATH500/AIME24/AIME25/AMC23分别计分，不以macro替代。完整精度、格式错误和截断数据在JSON中。
- 运行代码固定为`be2eff65325ed35d26911046beb4ab42d82bf8ad`，本次只归档，不修改已运行代码。

配置与数据详情见[冻结方案](../plans/2026-09-23-qwen17-base-grpo-blockfirst.md)和
[原启动记录](2026-09-23-qwen17-base-grpo-pair-startup.md)。

## Avg@8

单位为百分比。初始学生是未进行本轮训练的同一Base模型。

| 方法/步数 | MATH500 | AIME24 | AIME25 | AMC23 |
|---|---:|---:|---:|---:|
| 初始学生 | 44.40 | 2.50 | 0.42 | 17.62 |
| Token 50 | 64.18 | 3.33 | 3.33 | 31.33 |
| Block3 50 | 63.28 | 7.92 | 5.42 | 30.42 |
| Token 100 | 65.93 | 9.58 | 5.00 | 32.68 |
| Block3 100 | 64.15 | 6.67 | 5.42 | 32.98 |
| Token 150 | 63.63 | 7.08 | 6.25 | 33.13 |
| Block3 150 | 62.98 | 6.25 | 3.33 | 33.28 |
| Token 200 | 64.33 | 8.33 | 5.83 | 31.33 |
| Block3 200 | 63.03 | 6.25 | 2.92 | 34.04 |

## Pass@8

| 方法/步数 | MATH500 | AIME24 | AIME25 | AMC23 |
|---|---:|---:|---:|---:|
| 初始学生 | 78.60 | 10.00 | 3.33 | 46.99 |
| Token 50 | 84.60 | 13.33 | 13.33 | 60.24 |
| Block3 50 | 84.20 | 26.67 | 13.33 | 62.65 |
| Token 100 | 86.00 | 33.33 | 13.33 | 55.42 |
| Block3 100 | 83.20 | 16.67 | 23.33 | 57.83 |
| Token 150 | 85.60 | 23.33 | 13.33 | 61.45 |
| Block3 150 | 85.60 | 23.33 | 13.33 | 57.83 |
| Token 200 | 86.00 | 26.67 | 16.67 | 65.06 |
| Block3 200 | 85.80 | 20.00 | 13.33 | 63.86 |

## 结果边界

两种方法在本次评测中的各保存步数均高于初始学生，但不能由此推导Block3优于Token。
预先指定的Step200上，Block3减Token的Avg@8差为：MATH500 -1.30、AIME24 -2.08、
AIME25 -2.92、AMC23 +2.71个百分点；Pass@8四榜均较低。
MATH500在四个配对步数上的Avg@8均低于Token，其他榜单的差异随步数变化。

这里只是一个seed的完整对照，尚未做配对置信区间分析或多seed复现，不能宣称统计显著。
尤其AIME每套仅30题，不应逐榜选择不同的最优checkpoint后作为主结果。
这也不是教师筛选门槛已被证明有效的证据：筛选证据不足不等于教师无训练价值，
验证筛选预测能力仍需预先固定规则及更多师生对。

## 健康与衔接

Block3训练在05:02:20结束，Step200评测在05:08:28启动，交接约6分08秒。
初始学生评测结束后，Token独立保存/恢复探针通过，正式训练在08:58:44启动，17:38:58结束。
九次完整评测及最终验收于20:40结束，21:23检查时四张GPU均空闲。

训练累计截断：Block3为656/6400=10.25%，Token为711/6400=11.109375%。
标量诊断未发现非有限值，训练日志未发现CUDA OOM。
两组退出阶段均有DataLoader worker被Killed的日志堆栈，不能描述为日志完全无异常；
两组训练退出码均为0，完整状态、6400条轨迹和后续评测均通过验收，原日志保留。
这项退出期告警的底层原因尚未单独定位。

Step200评测截断率为：

| 方法 | MATH500 | AIME24 | AIME25 | AMC23 |
|---|---:|---:|---:|---:|
| 初始学生 | 1.25 | 6.67 | 3.75 | 4.07 |
| Token | 3.20 | 17.50 | 15.00 | 9.49 |
| Block3 | 3.18 | 15.83 | 11.25 | 6.33 |

Block3相对Token的截断较少，但两种方法都高于初始学生；不能将有限梯度或较少截断
等同于推理能力普遍提高。

## 证据与备份

完整训练状态及原始训练/评测轨迹保留在ml2：

```text
/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260923v5_qwen17_base_grpo_blockfirst_seed21_ml2
```

本地结果、配置、验收、标量、位置诊断和图表快照：

```text
/home/tyf/paper/outputs/qwen17_base_grpo_pair_20260923/final_20260924
```

本地快照不包含模型权重或完整原始rollout；这些大文件尚只保留在ml2。
八份关键结果/状态/标量文件的SHA256已与远程逐一核对。
Git只保存轻量结果、验收与来源记录，不提交权重或大体积轨迹：

- [各benchmark完整结果](../../results/qwen17_base_grpo_pair_20260923/per_benchmark_results.json)
- [配对结果与差值](../../results/qwen17_base_grpo_pair_20260923/paired_comparison.json)
- [九次完整评测验收](../../results/qwen17_base_grpo_pair_20260923/evaluation_acceptance.json)
- [实际训练输入配对验收](../../results/qwen17_base_grpo_pair_20260923/paired_rollout_acceptance.json)
- [冻结运行清单](../../results/qwen17_base_grpo_pair_20260923/queue_manifest.json)
- [最终队列状态](../../results/qwen17_base_grpo_pair_20260923/queue_state.json)

关键SHA256：

```text
7831698a4331814de799962d3fd29f15d06ee2bfa33ae403ffc189b34ba022ab  per_benchmark_results.json
1c1db3db162aaf2c0c87daea2201cd8c8dd6d10ed4a91d08dbad880305a928a5  paired_comparison.json
527074535fac54bc3f0704fe5e9aa80aa13336584ae7b4fd6f3474bb9b45e5be  evaluation_acceptance.json
```
