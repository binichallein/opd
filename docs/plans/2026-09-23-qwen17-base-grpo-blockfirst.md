# 4B-GRPO到1.7B-Base完整对照 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 执行用户在能力测试后明确授权的“启动训练，和上次一样”，保持筛选证据不足的结论不变。

**Architecture:** 私有命名空间复用最近的完整配对队列，新增Base输入、EOS和历史mask核验。
不修改loss、数据、采样、评测脚本或既有运行。能力门槛例外仅允许本次已封存结果，不放宽工程验收。

**Tech Stack:** ml2四张A100 80GB、vLLM、verl/FSDP、冻结历史grader。

## 用户授权与固定设置

本轮用户已明确要求忽略此次能力筛选未通过并开始训练，延用最近的完整对照。
不重问已确认的保存策略：两组各200步、seed21、50/100/150/200完整状态全部保留、不删除。

- 教师`lllyx/Qwen3-4B-Base-GRPO`，学生`Qwen/Qwen3-1.7B-Base`，两组独立从原始学生初始化。
- 教师能力报告SHA256固定为`d8ebca3e3edea8f0af35c69c192dd20d5a5e414304099735276358fc35d9fab0`。
  其status仍为inconclusive、passed=false、failures仅continuation_gain；新增独立授权记录，不改原报告。
- 复用原DAPO-Math-17K文件及seed21抽样顺序，每步4题x8条轨迹=32条，学习率2e-6，PPO epoch=1。
- 最大prompt2048，response16384，temperature1、top_p0.9、top_k=-1；逐请求独立seed规则不变。
- Base训推统一裸题目+boxed指令+Solution:，不使用ChatML或think标签；原生EOS151643，无额外停止token。
- 保留原历史Block3 Mean的block联合PPO ratio和归一化，不将其改写成仅共享advantage的另一种方法。
- actor/ref micro-batch=1、rollout logprob micro-batch=4、vLLM显存利用率0.6，四卡同用。
- 每步记录全部rollout及历史过程诊断，包括师生entropy/位置分布、gap、top16 overlap/mass、
  overlap advantage、loss、梯度范数、长度/截断、block符号翻转等；每组结束自动生成诊断图。
- 顺序：Block3保存/恢复探针 -> 正式200步 -> 完整评测200/150/100/50 -> 初始学生完整评测 ->
  Token保存/恢复探针 -> 独立正式200步 -> 完整评测200/150/100/50。
- 每次评测MATH500/AIME24/AIME25/AMC23共643题、每题8次，共5144条，历史grader逐benchmark报告
  Avg@8、Pass@8、格式、截断；不合并macro，不因分数不理想自动改参数。
- 教师没有可靠的显式固定下载revision标记，因此launcher的teacher revision字段为空；
  模型身份由能力测试中已有的文件SHA256与来源证据固定，不伪造HF或ModelScope标记。

## Task 1: 独立配置与回归

1. 新增`tests/test_qwen17_base_grpo_pair.py`，先验证新入口缺失导致测试失败。
2. 新增`scripts/run_qwen17_base_grpo_pair.py`；严格核验指定报告及用户例外，保留CPU Ray预热。
3. `run_qwen17_instruct_pair.py`只增加默认不变的配置接点：prompt编码、EOS、初始化说明和授权元数据。
4. 测试模型身份、顺序、保存策略、prompt/停止/mask、报告防篡改，以及旧指令队列不受影响。

## Task 2: 部署与GPU验收

1. 新commit与不可变`deployments/<commit>`，完整运行文件哈希校验；只连接ml2。
2. 本轮目录`runs/20260923v4_qwen17_base_grpo_blockfirst_seed21_ml2`，新tmpfs缓存`/dev/shm/q17g`。
3. 服务端nohup启动，记录PID/环境/命令；旧权重、结果和能力报告只读保护。
4. 验证真实collector对全部643道评测题与64道筛选题的输入一致性。
5. 首步保存，加载完整模型/优化器/调度器/随机数/数据位置继续第二步；门槛未通过即停。
6. 正式训练启动后核对首批源题目、seed、rollout、诊断、非有限值，不能只看队列PID。

## Task 3: 记录

提交启动记录至GitHub，小型证据备份本地；自动队列运行不依赖本地Windows开机。
每组末尾完整状态、轨迹与图均保留，最终结果另外归档，不提前宣称训练完成或方法有效。

## 执行状态

- [x] 用户授权、ml2四卡空闲、旧队列complete及模型文件来源核对。
- [x] 新旧CPU回归通过：新增12项先失败再实现，相关配对、能力筛选及completion核验共204项通过。
- [x] 不可变部署及nohup队列启动：19:28启动，643+64题输入核验和配对run card通过。
- [x] 第一步真实GPU更新、完整checkpoint及四rank优化器检查通过，32条轨迹核验通过；启动证据已备份本地。
- [ ] 从完整状态恢复第二步并通过恢复验收。
- [ ] 正式Block3训练启动并核验首批轨迹。
- [ ] 全部训练评测完成及最终归档。
