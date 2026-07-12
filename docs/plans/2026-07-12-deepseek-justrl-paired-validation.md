# DeepSeek / JustRL Block3 跨师生对复现实验契约

## 研究问题

在已经完成的 Qwen3-1.7B-Base 到 Qwen3-4B-Base-GRPO 实验中，
`block3_mean` 相对 sampled-token OPD 呈现小幅正向结果。本实验只检验一个问题：
当师生对改为同架构、同 tokenizer 的 DeepSeek-R1-Distill-Qwen-1.5B 与其
post-RL 版本 JustRL-DeepSeek-1.5B 时，这个方向能否复现。

这不是用新模型替代原始 baseline，也不比较两对模型的绝对分数。每对师生内部都以
sampled-token OPD 为 baseline，`block3_mean` 是唯一目标变量。

## 固定模型

- Student：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`。
- Student revision：`ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`。
- Student `model.safetensors` SHA-256：
  `58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945`。
- Teacher：`hbx/JustRL-DeepSeek-1.5B`。
- Teacher revision：`0637e4096c789c67f9eecbe8355e0bdeddede1c2`。
- Teacher `model.safetensors` SHA-256：
  `1bedcdf243c4e2e633fa04c05617809cf6a1bbc1a07221609035f6347efecffb`。
- 两者 tokenizer、词表、层数、hidden size 和 chat template 已在训练前核验一致。

## 固定训练配置

- 机器：`train`，4 张 A800-SXM4 80GB；两组顺序执行，不共享运行目录。
- 数据：原始 DAPO-Math-17K parquet pool，共 1,791,700 行。
- `train.parquet` SHA-256：
  `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`。
- 两组共用 seed 21、同一数据文件、同一采样脚本；诊断记录中的
  `prompt_batch_sha256` 用于检查相同步骤是否看到相同 prompt batch。
- 200 个 training steps，train batch 4，每个 prompt 采样 8 条 rollout。
- LR `2e-6`，最大 prompt 2,048 token，最大 response 16,384 token。
- rollout temperature 1.0，top-p 0.9，vLLM 显存比例 0.6。
- actor / rollout-logprob / reference micro-batch 分别为 `1 / 4 / 1`。
- 保存 Step 50、100、200 的完整 checkpoint，不自动删除中间里程碑。

两组唯一的目标函数差异：

- `token_opd`：每个采样 token 使用自己的 teacher-vs-old-student advantage。
- `block3_mean`：每 3 个连续采样 token 的 advantage 取均值，并让 block 内三个
  token 共用该信号。

## 训练前门禁

两个变体都必须分别通过：

1. Step 1 停止、完整 checkpoint 与诊断产物审计。
2. 从 Step 1 精确恢复到 Step 2，并再次审计完整 checkpoint、resume 路径和诊断。
3. 模型路径、Hub revision、数据哈希、不可变运行代码 commit 与子模块 commit 审计。

任一门禁失败时不启动正式训练。

## 诊断与热力图

- 每 5 steps 记录 student/teacher entropy、signed/absolute entropy gap、Top-16
  overlap、student/teacher overlap mass、overlap-token advantage、PG loss、裁剪前
  grad norm、response length 和 truncation ratio。
- 同时记录本方法专用的 sign-flip rate、weighted sign-flip rate、credit leakage、
  post-update block ratio、block/token advantage 分布和所有 non-finite 计数。
- Step 50、100、200 保存 position-level NPZ，并绘制 entropy、overlap、advantage、
  sign-flip、leakage 和 policy-drift 热力图。
- non-finite、overflow/underflow 计数大于零，或 checkpoint 不完整，均判为训练审计失败。

## 固定评测配置

- 评测 Step 50、100、200。
- Math500、AIME24、AIME25、AMC23 全量题集。
- 每题 `n=8`，rollout seeds 21-28，temperature 1.0，top-p 0.9，最大 16,384 token，
  thinking 开关关闭。
- 每个 checkpoint 必须产生 5,144 条 rollout，并通过题数、seed、数据哈希和模型路径审计。
- 保留 VERL 原始评分；主比较使用与历史 Qwen 对相同的外部 grader 重新评分，grader
  SHA-256 为
  `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。

## 比较和判定

- 主终点：Step 200 equal-task macro Avg@8 与 macro Pass@8 的
  `block3_mean - token_opd` 差值。
- 次终点：Step 50/100 的同口径差值、四个任务逐项差值、prompt 级 win/tie/loss。
- 两组评测必须在 `(task, example_id, rollout_id, seed)` 上精确配对。
- 使用 10,000 次 task-stratified prompt bootstrap，固定 bootstrap seed 20260712，
  报告 95% CI；不能只展示有利任务或有利 checkpoint。

预先固定结论口径：

- **跨师生对方向复现**：DeepSeek/JustRL 对的 Step-200 macro Avg@8 和 Pass@8
  点估计都大于 0，且与原 Qwen 对方向一致。
- **强复现**：在方向复现基础上，DeepSeek/JustRL 对两个主终点的 95% CI 都严格大于 0。
- **不复现**：任一主终点点估计小于等于 0。

本实验仍然只有一个 training seed。bootstrap 只量化固定 checkpoint 下的题目采样
不确定性，不能替代多 training-seed 方差。
