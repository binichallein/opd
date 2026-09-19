# Qwen completion 与独立请求 seed 验收

用户授权验证三个点；不恢复已经停止的 v1 队列，不自动启动正式 200 步。

## 1. 原始师生的最终提示

- 学生：ModelScope 官方 Qwen3-4B-Base，既定固定 revision 与哈希。
- 教师：原实验公共 Qwen3-4B-Base-GRPO，使用历史资产清单核验，非私人模型。
- 题目：停止的 Qwen v1 正式训练 Step1--4 原始轨迹中的 16 题。
- 对照：历史 ChatML；裸题目加 `Solution:`；最终 completion 提示。
- 每题 2 次，seed21/22；两模型三提示合计 192 输出。
- BF16、TP1、温度1、top-p0.9、top-k-1、最大16384 token，原模型 EOS151643。
- 保存实际输入输出 ID、logprob、终止原因、版本、输入和输出哈希。
- 使用固定历史 grader；正确率、缺少 boxed、重复、截断分别统计。
- 这是固定训练题诊断，不是正式 benchmark。独立生成的教师正确率不足以证明
  它在所有学生前缀上可靠，后续短训练另检查实际师生分布与梯度。

最终提示使用共享函数，训练和评测不得各自拼接：

```text
{question}

Please solve the problem step by step and put the final answer in \boxed{}.

Solution:
```

最后保留一个换行。没有 ChatML、think 标记或额外 BOS。

## 2. 真实训练接线

- 新协议 `qwen3_completion_boxed_v1`；历史协议、损失、EOS mask 均保持原样。
- 请求种子：全局21、step、数据行编号、原题 SHA256、rollout 序号、turn 组成
  规范 JSON，SHA256 前8字节取非负63位整数；不依赖方法名、进程、GPU、UUID。
- 在分片与填充前赋予请求身份；每条 vLLM 请求使用自己的 SamplingParams。
- 全量归档真实 sampling seed 和身份。发现缺失身份/不支持的动态过滤时拒绝运行。
- 评测仍保留历史 seeds21..28；新请求 seed 只改变训练重复采样问题。

## 3. 有界 GPU 更新与恢复

- 两组独立从原始学生启动；Block3 Mean 保留历史 joint-block ratio 和损失归一化。
- 相同数据、4题×8轨迹、LR2e-6、16K上限、microbatch1、vLLM0.6、4张A100。
- 每组只做 Step1 保存、退出、恢复至 Step2；不将探针权重用于正式训练。
- 检查四 rank 优化器/调度器/RNG、dataloader游标、恢复后的题目与 seed。
- 保留每步全部轨迹、原有监控标量及位置热图；运行结束检查 GPU 进程退出。
- 验收区分工程通过与科学有效：种子独立、恢复正常不等于已经证明 Block3 涨分。
