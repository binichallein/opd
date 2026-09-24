# 1.7B / 4B-GRPO：单回答、100步完整方法对照

用户纠正：不是B/C消融，而是在新采样配置下重做原师生对，检查性能提升能否复现。
只准备与核对配置，沿用上一条“先报告、不急着开始训练”的限制，不启动GPU任务。
先前B/C准备目录保留为历史记录，但已被本方案取代，禁止启动旧B/C命令。

## 固定配置

| 项目 | 本轮设置 |
|---|---|
| 学生 | 官方Qwen3-1.7B-Base，原始权重 |
| 教师 | 公开lllyx/Qwen3-4B-Base-GRPO，原有同一权重 |
| 对照组 | Token OPD：block size=1，sum，legacy loss |
| 实验组 | 历史完整Block3 Mean：block size=3，mean，legacy loss |
| 每组更新步数 | 100 |
| 每步prompt位置数 / 每题回答数 | 32 / 1 |
| 有效轨迹数 / PPO mini-batch | 32 / 32 |
| 数据 | 同一DAPO-Math-17k train.parquet，SHA不变 |
| seed | 21；仍保留逐请求seed21，不同时修改采样seed规则 |
| 学习率 / PPO epochs | 2e-6 / 1 |
| max prompt / response | 2048 / 16384 |
| temperature / top-p | 1.0 / 0.9 |
| GPU / actor micro-batch / vLLM显存比例 | 4 / 1 / 0.6 |
| 保存节点 | Step50、100，模型、优化器、scheduler、RNG和数据位置完整保存，不删除 |
| 过程记录 | 全部raw rollout，原每5步诊断、位置统计及后续热力图 |

两组从原始学生独立初始化，不在已停止的B Step50上接续，不互相继承训练权重。
总步数改成100，因此scheduler配置的总训练步数也为100，不是继续一个200步任务。

完整Block3同时包含共享均值advantage、联合block PPO ratio和历史block归一化。
两组`opd_block_ablation`均必须为`legacy`；不是`adv_only`或`joint_tokenmean`。
保留冻结训练代码`7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5`，不修改loss实现。

## Prompt保持旧版

训练仍为`qwen3_historical17_v1`：原ChatML与以下数学指令，不替换为裸题completion：

```text
Math problem: {question}

Please carefully reason through the math problem step by step and derive the correct answer. You must conduct reasoning inside <think> and </think> and give the final answer within \boxed{}.
```

评测仍为原历史非thinking chat输入；保留已知的训推提示差异，不借本次实验修复它。
旧EOS、stop token及mask行为也不改变。这个历史复现约束不是推荐的新模型通用模板。

## 比较与评测

计划完整评测原始学生、Token Step50/100、Block3 Step50/100，共五个模型状态。
沿用历史grader及相同采样/长度设置，每题8条；分别报告MATH500、AIME24、AIME25、AMC23
的Avg@8和Pass@8，不混合题目计算总分。保留全部评测输出和格式/截断等健康指标。
当前仅生成评测协议记录，没有启动评测。

主要判断同一步数的Block3是否优于新Token对照；其次比较两者相对初始学生的提升。
只超过初始学生，不能证明Block3优于Token；只有seed21，也不能声称跨seed稳定。
历史Step50/100可作为背景，但本轮每100步有3200次prompt出现，旧4x8在100步仅400次，
两者均3200条轨迹，所以这是改变采样覆盖后的复验，不是旧实验的逐样本重放。
数据文件包含重复题，32个prompt位置不保证每步有32道去重后的不同题。

## 运行与校验

- 准备器：`scripts/prepare_historical_pair_n1.py`，没有训练启动分支。
- ml2新目录：`runs/20260925v2_historical17_pair_n1_step100_seed21_ml2`。
- 只复用旧B run card中的非loss设置来约束差异，不复用其消融方法。
- 两组实际run card只允许方法及输出标识不同；数据/运行代码manifest必须完全相同。
- 旧Step50及独立恢复Step51不改动；此前32x1 B/C配置只添加被取代说明，不覆盖证据。
- 正式启动前仍需适配新采样配置的输入/保存恢复验收，不能把旧B恢复探针当成新两组的验收。
