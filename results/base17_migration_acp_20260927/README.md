# Base1.7B / GRPO4B 剩余任务迁移到 ACP

## 用户授权与执行顺序

2026-09-27，用户要求把 ml2 尚未执行的评测与 Token OPD 实验迁往四张 H100 的 ACP。
不是重跑 ACP 的指令版消融，也不重训已完成的 Block3。

1. 复制并校验 ml2 的 Block3 四个完整 checkpoint、原始轨迹、诊断、模型和来源记录。
2. 在 ACP 完整评测 Block3 Step100、75、50、25。
3. Token OPD 从原始 Qwen3-1.7B-Base 独立初始化，先验证保存第1步及恢复至第2步，正式训练100步。
4. 在 ACP 完整评测 Token Step100、75、50、25，按 benchmark 分开汇总。

## 不变的实验条件

- 教师为原公开 Qwen3-4B-Base-GRPO，学生为原官方 Qwen3-1.7B-Base，按源文件哈希锁定。
- 相同 DAPO 文件和3200个题目位置及顺序；每步32题，每题1条，seed21，保留每请求seed21。
- lr2e-6，原 Token/Block3 loss，不改变 batch、更新次数、rollout 采样温度或响应上限。
- 裸题目、boxed 指令、Solution:；不用 ChatML/think，原生 EOS151643，训推协议一致。
- 保留25/50/75/100全部完整训练状态，不删除权重；不重评初始学生。
- MATH500、AIME24、AIME25、AMC23，643题各生成8条，使用同一历史 grader。
- 保存全部训练/评测 rollout、过程指标及位置热力图。

## 必须记录的迁移差异

Block3已在 ml2 的4张 A100上训练；Token改在 ACP的4张 H100上训练。两组本轮评测都在ACP。
ACP CPU配额32，ml2原Ray CPU数64；Python构建分别为3.12.3与3.12.13。
实际检查两端Torch均为2.8.0+cu128、CUDA12.8，不把它误记为版本变化。
因此这是跨硬件的延续实验，不声称同硬件因果隔离或逐比特复现。

ACP另建 `envs/verl-cu128-base17-migration-v1`，不修改已有指令版实验使用的环境。
数学解析依赖按ml2匹配：antlr4-python3-runtime4.7.2、mathruler0.1.0；其他关键
模型、数学评分、数据和Ray版本也在启动前校验。不静默接受grader或依赖漂移。

## 数据保护与自动执行

- ml2源run：`runs/20260926v1_qwen17_base_grpo_protocol_n1_seed21_ml2`。
- ACP导入：`/mnt/afs/202609/tyf-qwen-opd/imports/20260927_ml2_base17_protocol`。
- ACP新run：`/mnt/afs/202609/tyf-qwen-opd/runs/20260927v1_base17_grpo_migrated_n1_seed21_acp`。
- 源代码固定为3316918；本轮只新增迁移控制器，不改训练、损失或评测实现。
- `migration_manifest.json`记录2355个文件，共101849693103字节（含ACP已有且须匹配的数据）。
- 新传输在ACP后台独立运行，直接连接ml2；不复制用户私钥，不依赖Windows中转。
- 初次逐目录SCP被主动停止，未完成副本保留在`.migration_part`目录。当前单连接流式传输
  使用`.stream_part`文件，逐文件校验后才更名；最终全部校验后才生成`transfer_acceptance.json`。
- `wait_for_verified_migration.py`等待绑定manifest哈希的完整验收，再只启动一次新队列。
  失败保留现场，不自动重试，不覆盖旧run，不热改正在使用的代码。
- Token必须通过独立保存/恢复验证才会正式训练；所有产物放AFS，仍受ACP平台时限约束。

## 状态说明

本文是迁移配置与来源记录，不是评测完成证明。实时状态以ACP的
`transfer_state_v2.json`、新run的`migration_wait_state.json`、`queue_state.json`
及`evaluation_acceptance.json`为准。迁移开始时，ml2只有Block3训练完成，尚无本轮benchmark分数。
已提交的论文与已有ACP实验均未修改。

## 01:12 北京时间启动验收

- 冻结版本：`e07997afe6cb65b803de46486d1482ffe0d3e38d`。
- ACP实际环境170项测试通过；本地迁移与等待专项61项通过，扩展不依赖Torch的测试148项通过。
  本地默认Conda的Torch库存在`iJIT_NotifyEvent`导入问题，因此相关13项测试转到ACP实际环境验证，全部通过。
- ACP独立新环境16项数学/数据/Ray依赖与ml2匹配，绘图可导入；旧ACP环境保持原样。
- 接收PID336441，等待队列PID339663，均PPID1；建立传输时用的本地临时SSH agent已经退出。
- 已校验25549594264字节（含ACP已有且哈希匹配的数据文件），总清单101849693103字节。
  此时**传输未完成，GPU评测尚未开始**，不把等待队列启动说成评测启动。
- `startup_20260927.json`、`environment_preparation.json`和`runtime_tests.json/.log`保留实际验收证据。
