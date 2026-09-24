# 1.7B / 4B-GRPO：单回答、100步完整方法对照

用户纠正：不是B/C消融，而是在新采样配置下重做原师生对，检查性能提升能否复现。
用户随后授权启动：先Block3训练及评测，之后Token训练及评测。
最新补充要求每25步保存一次，共四个权重；两组都遵循相同规则。
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
| 保存节点 | Step25、50、75、100，模型、优化器、scheduler、RNG和数据位置完整保存，不删除 |
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

最新要求：原始学生已有评测，不重复评测。本轮仅完整评测Token Step25/50/75/100、
Block3 Step25/50/75/100，共八个训练权重。
沿用历史grader及相同采样/长度设置，每题8条；分别报告MATH500、AIME24、AIME25、AMC23
的Avg@8和Pass@8，不混合题目计算总分。保留全部评测输出和格式/截断等健康指标。
顺序：Block3独立保存/恢复探针 → 正式100步 → 评测100/75/50/25
→ Token独立保存/恢复探针 → 原始学生初始化正式100步 → 评测100/75/50/25。
任一步失败即停止，不自动跳过或改参重试。

主要比较Step100，另外完整展示25/50/75，判断Block3是否优于匹配的Token对照；
其次比较两者相对初始学生的提升，不依据结果挑选最佳checkpoint作为唯一结论。
只超过初始学生，不能证明Block3优于Token；只有seed21，也不能声称跨seed稳定。
历史Step50/100可作为背景，但本轮每100步有3200次prompt出现，旧4x8在100步仅400次，
两者均3200条轨迹，所以这是改变采样覆盖后的复验，不是旧实验的逐样本重放。
数据文件包含重复题，32个prompt位置不保证每步有32道去重后的不同题。

## 运行与校验

- 准备器：`scripts/prepare_historical_pair_n1.py`，没有训练启动分支。
- 自动队列：`scripts/run_historical_pair_n1.py`，服务器端nohup运行，不依赖本地连接。
- ml2新目录：`runs/20260925v4_historical17_pair_n1_step100_save25_seed21_ml2`。
- v2仅50/100的配置保留但不启动；新控制版本不修改冻结训练部署。
- v3准备快照写于取消初始评测之前，保持原样；队列manifest单独记录取消初始评测的修订，
  实际自动顺序不包含原始学生。不能因旧准备快照写了include_initial而重复启动它。
- v3在CPU Ray预检因嵌套socket路径109字节超过107限制停止，没有GPU训练/探针。
  v4缩短Ray gate临时路径，添加回归测试及启动前检查；原v3全部记录保留不重跑。
  v4准备记录也已取消初始评测；loss、模型、数据、prompt及所有训练超参数不变。
- 只复用旧B run card中的非loss设置来约束差异，不复用其消融方法。
- 两组实际run card只允许方法及输出标识不同；数据/运行代码manifest必须完全相同。
- 旧Step50及独立恢复Step51不改动；此前32x1 B/C配置只添加被取代说明，不覆盖证据。
- 正式启动前仍需适配新采样配置的输入/保存恢复验收，不能把旧B恢复探针当成新两组的验收。
- 每组探针保持100步scheduler配置，先保存Step1，退出，再恢复保存Step2。
  验证四rank优化器、scheduler、RNG、数据位置及Step1原文件不变，正式不继承探针权重。
- 用冻结代码的实际sampler预先记录100步输入计划，核对每步32条、每题1条及逐条来源。
  保留prompt/seed指纹，训练完成后逐步比较两组输入；不要求更新后的输出相同。

## 历史准备验收（v2，已被四权重配置取代）

2026-09-25 01:48北京时间，ml2两份实际run card与命令核对通过，均100步、32x1、
legacy loss；两组只有方法及输出标识不同，数据/代码manifest一致，bash语法检查通过。
42项本地相关测试通过。四张GPU空闲，没有训练/评测进程启动，没有新增权重或rollout。
控制脚本commit为`6dab418fc9fafbd443413ce31b984ce2a1355ef0`，训练部署仍为原7bf5420。
旧单回答B/C目录已添加`superseded.json`，原文件未覆盖；旧checkpoint不改动。

本地配置与验收备份：`/home/tyf/paper/outputs/historical17_pair_n1_step100_20260925`。
Git精简备份：`results/historical17_pair_n1_step100_20260925/`。
`preparation.json`记录实际配置和评测协议，`preparation_acceptance.json`记录校验及未启动状态。
