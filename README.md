# OPD 内部实验仓库

这是一个内部研究仓库，用来同步、规范和复现 On-Policy Distillation
（OPD）相关实验，重点是语言模型后训练中的 token-level / block-level
reverse-KL 目标。

当前仓库包含 clean-room OPD 训练与评测脚本、block supervision 单测、
整理后的结果摘要，以及 block reverse-KL 实验的 HTML 报告。
同时，`external/revisiting_opd` 固定了公开 `revisiting_opd` codebase，
用于后续论文级 baseline 与 DAPO-Math-17K 对齐实验。

## 当前授权：完整 Block3 优先，100步，每25步保存

用户已纠正范围：不是B/C消融，而是重做`Qwen3-1.7B-Base`学生与公开
`Qwen3-4B-Base-GRPO`教师的Token OPD / 历史完整Block3 Mean对照。
每组100步，每步32个prompt、每题1条，保存Step25/50/75/100完整状态，不删除；
prompt、seed21及其余设置不变。先Block3训练及四个权重完整评测，
再Token独立训练及四个权重完整评测，各benchmark单独计分。
初始学生已经评测过，按最新要求不重复评测；本轮只新增八次权重评测。
[本轮方案](docs/plans/2026-09-25-historical17-pair-n1-step100.md)，
准备器`scripts/prepare_historical_pair_n1.py`生成配置，
队列`scripts/run_historical_pair_n1.py`在各组保存/恢复验收后启动正式任务。
当前目录`runs/20260925v4_historical17_pair_n1_step100_save25_seed21_ml2`，以LIVE状态为准。
v3在CPU阶段因Ray socket路径过长停止，未进行GPU训练；v4仅缩短临时目录，原记录保留。
[启动验收及留存证据](docs/results/2026-09-25-historical17-n1-save25-startup.md)：
截至03:17北京时间，Block3保存/恢复验收通过，正式训练已完成Step3/100。
前两步64条正式轨迹已核验并本地备份；梯度有限，但截断和重复仍偏高，不能称为生成健康。
评测尚未开始，保存25/50/75/100及后续Token流程不变；此处是时点记录，以LIVE状态为准。
旧B/C及v2仅保存50/100的准备均已被取代，保留证据但不要启动。

13:18只读更新：Block3训练、四份状态及3200条轨迹已验收，Step100/75完整评测完成，
正在Step50评测，Token尚未开始。[长尾排查](docs/results/2026-09-25-historical17-n1-eval-long-tail.md)
确认重复生成集中于部分评测seed，静态整轮分片放大等待；未更改本轮seed或停止条件。

## 历史记录：旧版 B/C 已停，单回答准备已被取代

2026-09-25按用户要求停止旧版1.7B-Base / 4B-GRPO的B训练及自动队列，
最后完成Step56，完整保存节点Step50，C未开始。
[停止、恢复验收与新配置方案](docs/plans/2026-09-25-historical17-single-rollout.md)：
新B/C仅把每步4题x8回答改为32题x1回答，仍200步、PPO batch32，
seed21、旧prompt、loss、数据、checkpoint和观测配置不变。
`scripts/prepare_historical_single_rollout.py`只生成配置，不启动训练或评测。
等待用户明确启动；禁止沿用下方历史自动启动授权重启旧队列。
[最终验收与留存](docs/results/2026-09-25-historical17-stop-n1-prepared.md)：
Step50实际恢复到51并保存/退出，原checkpoint全部22个文件SHA不变，四卡空闲。

## 已完成对照：4B-GRPO到1.7B-Base

2026-09-24 20:40北京时间，v5队列全部完成：两组各200步、八个训练权重及初始学生
共九次完整评测均验收通过。[完整结果与备份](docs/results/2026-09-24-qwen17-base-grpo-pair-final.md)。
Step200的Block3相对Token在MATH500/AIME24/AIME25的Avg@8较低，AMC23较高；
四榜Pass@8均较低。本轮不支持Block3普遍优于Token，不能仅凭初始学生对比宣称方法胜出。

教师能力筛选结果为证据不足；用户随后明确要求“这次不管了，启动训练吧，和上次一样”。
[本轮完整对照方案](docs/plans/2026-09-23-qwen17-base-grpo-blockfirst.md)不修改原筛选结论，
以单独授权记录继续，保存/恢复、输入和完整评测的工程验收仍必须通过。
先Block3 Mean训练200步及200/150/100/50评测，再初始学生评测，最后Token OPD独立训练和四次评测。
两组seed21、DAPO文件及顺序相同，Base统一completion输入，保留全部恢复状态、rollout和诊断图。
原v4两步恢复探针已通过，正式首步前卡在Ray驱动注册；旧现场完整保留。
20:26启动独立恢复入口`scripts/recover_qwen17_base_grpo.py`，新目录为
`runs/20260923v5_qwen17_base_grpo_blockfirst_seed21_ml2`。
[故障与恢复记录](docs/results/2026-09-23-qwen17-base-grpo-ray-recovery.md)：
只关闭Ray worker预启动等待，不改变实验设置；实际进度以新目录状态及正式rollout为准。
[原启动记录及本地证据](docs/results/2026-09-23-qwen17-base-grpo-pair-startup.md)保留历史时点。

## 已完成能力测试：4B-GRPO到1.7B-Base

能力测试阶段的用户授权为只筛选教师，训练另定。教师为历史公开
`lllyx/Qwen3-4B-Base-GRPO`，学生为官方`Qwen/Qwen3-1.7B-Base`。
主测试使用Base裸题目、boxed指令和`Solution:`，教师接收学生原始输入和续写前缀；
另测教师原生chat对照。只读已有模型，不修改权重或启动训练。

- [固定测试协议与实施计划](docs/plans/2026-09-23-qwen17-base-grpo-teacher-screen.md)
- [最终结果与完整本地备份](docs/results/2026-09-23-qwen17-base-grpo-screen-final.md)：18:59完成，教师独立42/128对学生13/128，但续写12/64对13/64，结果为证据不足；健康门槛均通过，未启动训练。
- [启动记录与本地证据](docs/results/2026-09-23-qwen17-base-grpo-screen-startup.md)保留启动时点状态，不代表最终结论。
- 推理入口：`scripts/run_qwen17_base_grpo_screen.py`；资格判断复用既有预设门槛。
- 新运行目录：`runs/20260923v3_qwen17_base_grpo_teacher_screen_ml2`。
- 512条正式输出，另8条短检查；保留全部原始轨迹，不根据结果改参重试。
- GPU启动与最终验收以该目录的`queue_state.json`为准；测试通过也不自动训练。

## 已完成：0.6B学生的八组教师筛选

[冻结方案](docs/plans/2026-09-23-qwen06-teacher-screen.md)：对用户指定的4B/8B
Base、官方指令版及公共4B-GRPO教师进行能力诊断，本轮不训练。
主验收使用学生实际的完整输入和续写前缀，教师原生模板独立答题作为对照。
复用64道DAPO题、历史grader、配对bootstrap和预设健康门槛；原始rollout全部保存。
八对共享的学生基线不重复计为独立证据，诊断不代替完整benchmark评测。

- `scripts/run_qwen06_teacher_screen.py`：ml2四卡独立推理队列，失败不自动改参重跑。
- `scripts/qualify_qwen06_teachers.py`：逐token输入核对、原生模板对照去重和分对报告。
- `configs/experiments/qwen06_teacher_screen_assets.json`：ModelScope固定revision与SHA256。
- 运行状态以 `runs/20260923v1_qwen06_teacher_screen_ml2/queue_state.json` 为准。
- [最终验收结果](docs/results/2026-09-23-qwen06-teacher-screen-final.md)：02:51全部完成，27个单元核验无错误；官方指令版4B和8B到0.6B两对通过预设门槛，另六对未通过。未启动训练。
- [启动验收与本地原始证据](docs/results/2026-09-23-qwen06-teacher-screen-startup.md)保留历史启动时点记录。
- GRPO教师自身用过DAPO，本轮不能宣称未见数据泛化或筛选规则已能预测蒸馏收益。

## 已启动队列：4B到0.6B指令版配对训练

用户选定官方`Qwen/Qwen3-4B`教师和`Qwen/Qwen3-0.6B`学生，
[复用8B到1.7B指令版对照方案](docs/plans/2026-09-23-qwen06-instruct-blockfirst.md)。
先Block3 Mean、再Token OPD，各200步，保存50/100/150/200和全部恢复状态；
每组训练后自动完整评测四个权重，另评测原始学生，逐benchmark计分。
`scripts/run_qwen06_instruct_pair.py`在独立命名空间复用既有队列，不覆盖旧运行；
新增教师门禁和CPU Ray预热，损失、数据、prompt与采样不变。
实际状态以`runs/20260923v2_qwen06_instruct_blockfirst_seed21_ml2/queue_state.json`为准，
CPU测试通过不代表GPU恢复门禁或正式训练已通过。
v1在任何GPU更新前因缺少revision标记停止，现场保留；v2仅补齐已核验资产的来源元数据。
[启动记录](docs/results/2026-09-23-qwen06-instruct-startup.md)：03:32:48启动v2后台队列，
03:35:50开始Block3小步保存/恢复验收；正式200步需等待门禁通过。

## 已完成：指令版1.7B对8B的完整配对评测

2026-09-22 20:29北京时间，原始学生及两组各Step50/100/150/200的9次完整评测
全部验收通过。[最终分项分数、对照配置与归档入口](docs/results/2026-09-22-qwen17-instruct-final.md)。
Step200 Block3的四项Avg@8略高于Token，但Pass@8及其他checkpoint有胜有负，
不能概括成稳定全面提升。只有一个训练seed。

以下保留启动时的方案；原队列Token启动失败后，通过独立恢复队列完成，旧产物不覆盖。

教师采用已通过验收的官方 `Qwen/Qwen3-8B`，学生为官方 `Qwen/Qwen3-1.7B`。
先Block3 Mean训练200步，再评测200/150/100/50及原始学生；随后独立从原始学生
训练Token OPD200步并评测四个权重。两组均保留完整50/100/150/200状态与每步轨迹。

- [本轮配置、对照边界与实施进度](docs/plans/2026-09-21-qwen17-instruct-blockfirst.md)。
- 使用已验收的原生chat、显式关闭thinking，训推输入逐token核验；不使用旧Base提示。
- 相同DAPO文件、seed21、采样配置和历史loss；四个benchmark分别计分，保留全部评测输出。
- 控制器 `scripts/run_qwen17_instruct_pair.py`：恢复门禁失败或任何验收失败即停止，不自动改参重跑。
- 最终状态以 `runs/20260922v1_qwen17_instruct_token_recovery_seed21_ml2/queue_state.json` 为准。
- 23:38后台队列已启动，先做预检和恢复门禁；[启动证据与验证范围](docs/results/2026-09-21-qwen17-instruct-blockfirst-startup.md)。

## 已完成：8B对1.7B的两组能力诊断

2026-09-21用户要求先测官方 `Qwen3-8B-Base -> Qwen3-1.7B-Base`，再测
`Qwen3-8B -> Qwen3-1.7B` 指令版（官方仓库不带`-Instruct`后缀）。
本轮只做推理验收，不启动OPD训练。

21:40两组均已结束：Base独立正确率师生均13/128，未过能力门槛；
非thinking指令组独立38/128至60/128，续写18/64至32/64，通过预设验收。
指令教师续写仍有6/64截断、3/64周期重复，后续训练需要保留这些风险的监控。
[完整分项结果与解释边界](docs/results/2026-09-21-qwen8-to17-final.md)。

- [冻结设计与实现计划](docs/plans/2026-09-21-qwen8-to17-diagnostics.md)
- 复用上一轮64道诊断题、每题2次；各组取自己学生的前32题前缀做配对续写。
- Base使用completion，指令版使用原生chat且显式关闭thinking；先做真实GPU输出检查。
- 768条诊断与8条短输出检查全部保存，不将诊断分数冒充完整benchmark评测。
- 新队列：`scripts/run_qwen17_pair_diagnostics.py`，ml2唯一执行；Base结果不阻断指令组测试。
- 能力通过也不会自动训练；状态以新目录的`queue_state.json`为准。

## 已完成：8B对4B教师能力验收

旧队列于17:36结束，8B-Base没有通过能力门槛，正式训练未启动。
独立答题学生16/128、8B13/128、旧GRPO42/128；续写分别11/64、12/64、16/64。
最终状态为证据不足（inconclusive），不是程序失败。
[现场巡检和产物路径](docs/results/2026-09-21-qwen8-teacher-supervision.md)。

以下保留该轮启动时的条件计划，不代表当前仍会运行：

2026-09-21，用户选定官方 `Qwen/Qwen3-8B-Base`，学生继续使用原始
`Qwen/Qwen3-4B-Base`。先做固定题目独立解题及学生前缀续写验收；只有能力与
更新/恢复门禁均通过，才启动Token OPD和Block3 Mean各200步。不能默认参数更大
就比原4B-GRPO教师更强，也不把本次诊断分数当成完整benchmark评测。

- [验收标准、训练对照和实现计划](docs/plans/2026-09-21-qwen8-teacher-acceptance.md)
- [启动状态、PID、数据预检与工程验证](docs/results/2026-09-21-qwen8-teacher-startup.md)：
  16:14后台队列已启动，先下载/验收教师，不代表训练已开始或教师已合格。
- 控制器：`scripts/run_qwen8_teacher_pair.py`；仅ml2，新目录，不修改历史产物。
- 同一数据、seed21、completion提示、原始loss；全部50/100/150/200状态、每步轨迹和热图数据保留。
- 状态以新队列的 `queue_state.json` 为准；未通过验收不得启动正式训练。

## 已完成：4B对照、8B到1.7B指令版与ICLR2027双语论文

2026-09-23，中英文论文新增已验收的官方Qwen3-8B到Qwen3-1.7B指令版结果，
保留初始学生和两方法各50/100/150/200步的全部分项评分、曲线及行为指标。
Step200四项Avg@8差值为正，但配对题目区间均包含零，Pass@8并非一致提高；
不能据此宣称稳定优于Token OPD或多seed复现。

- [本次论文更新、证据边界及交付记录](docs/results/2026-09-23-paper-qwen17-instruct-update.md)
- [新增指令版分项指标](paper/iclr2027/generated/qwen17_instruct_20260923/qwen17_results.csv)
- [紧凑证据、配对区间及来源哈希](paper/iclr2027/generated/qwen17_instruct_20260923/qwen17_summary.json)

2026-09-21，Qwen3-4B-Base Token OPD的Step200、150、100、50完整评测均已验收，
与已完成的Block3对照逐benchmark比较。没有继续启动其他训练。
Block3的表现随checkpoint及任务变化，不能概括成稳定涨点。

- [最终分项结果、来源和数据重叠审计](docs/results/2026-09-21-qwen4-evaluation-and-paper-final.md)
- [中英文论文源码、图表与构建说明](paper/iclr2027/README.md)
- [全部当前分项指标](paper/iclr2027/generated/qwen4_publication/qwen4_results.csv)
- [单训练对条件下的配对题目区间](paper/iclr2027/generated/qwen4_publication/qwen4_paired_ci.csv)

以下条目保留各历史时点状态，不代表当前仍有任务运行。论文内部来源索引不得加入匿名投稿包。

## 新增 Llama 3.2 配对验证

2026-09-18 用户选择 `Llama-3.2-1B-Instruct <- Llama-3.2-3B-Instruct`，两者从
ModelScope 固定 revision 获取。先等当前 Qwen06 Block3 三个 checkpoint
完整评测成功结束，再执行零步 Instruct 学生评测、Token OPD 训练/评测、
Block3 mean 训练/评测。两臂独立使用同一初始学生，保留同一 DAPO 文件、
seed21、200步、全部监控及50/100/200完整状态与四任务 n8 历史评分器。

- [执行计划和完整配置](docs/plans/2026-09-18-llama32-paired-validation.md)
- [准备、测试及后台等待证据](docs/results/2026-09-18-llama32-preparation.md)：
  02:32北京时间，PID297837、runtime `f2d148e`，等待当前Qwen完整评测，尚未占用GPU。
- `scripts/prepare_llama32_assets.py`：ModelScope 原始 BF16 权重及26文件校验。
- `scripts/run_llama32_pair.py`：等待前驱、GPU prompt门禁、恢复门禁、串行实验。
- Llama 使用原生模板/停止符、固定日期和显式 token IDs；不注入 Qwen think
  标记。数据、loss和grader不变。新部署与旧实验隔离，不抢卡或自动重试。
- 每个benchmark独立报告 Avg@8、Pass@8、缺boxed比例与真实引擎截断率。

下面的0.6B/窗口运行状态保留历史时点记录；最新状态以服务器queue_state为准。

## 新增 0.6B 配对验证

2026-09-13 用户批准仅先做官方 `Qwen/Qwen3-0.6B-Base`，教师仍为原来
公共 `Qwen3-4B-Base-GRPO`。顺序固定为 Token OPD 完整训练/评测，再做
原版 Block3 mean；两组从同一 Base 独立初始化，仅改变监督方法。
同一 DAPO 文件、seed21、200 steps，Step50/100/200 四任务全量 n8，
历史 grader、监控和热图不变。等待旧窗口队列完成；用户后续已取消48小时
硬停限制，仍只执行这两组各200步及完整评测，不自动追加实验。

- [完整对照合同与执行计划](docs/plans/2026-09-13-qwen06-paired-validation.md)
- [部署与后台等待证据](docs/results/2026-09-13-qwen06-pair-startup.md)：最新PID3547500，
  runtime `ec0a7a9`，已取消时间上限，当前等待旧评测队列完成，尚无0.6B新得分。
- [HTML 对照表](reports/block_opd_experiment_report.html#qwen06-pair-20260913)
- `scripts/run_qwen06_pair.py`：限定 ml2、前驱完成门禁、配置审计、恢复验证、串行队列。

这不是早期 0.6B Instruct clean-room pilot，也不代表已有 0.6B 配对结果。

**最新计分要求（2026-09-13）：四个benchmark分别计分、分别报告。**
MATH500、AIME24、AIME25、AMC23各自列出Token/Block3的Avg@8、Pass@8与差值，
不合并成总分，也不用macro代替。Step50/100/200都按此方式展示。底层分任务
评分记录已存在；以下旧实验的macro记录仅保留作历史证据，不是新的报告口径。

## 本轮窗口实验

2026-09-11 的受限算力实验仅在 `ml2` 执行 `random3` 和 `sliding3`，
各 training seed21、200 steps。两组训练均已完成；截至 2026-09-12 23:40
北京时间，评测读取故障已修复，Random3 三轮历史重评完成，Sliding3 正在
补齐完整评测。保留 Step50/100/200 四任务全量 `n=8`、监控与热图；
最终配对结论尚待结果，不自动追加其他方法或 seed。

- [固定方案与预算](docs/plans/2026-09-11-sliding-window-opd-validation-v2-seed21.md)
- [启动证据、PID、目录和查询命令](docs/results/2026-09-11-window-seed21-startup.md)
- [JSONL 审计修复与评测恢复](docs/results/2026-09-12-window-eval-recovery.md)
- [后续研究草案：归一化信用分配与跨师生/数据验证](docs/plans/2026-09-12-post-window-research.md)
- [HTML 报告](reports/block_opd_experiment_report.html#window-recovery-20260912)

runtime 固定为 `fbad852a638de18e20d571a061b2be0437942a38`；之后的文档提交
不改变训练代码。下文的旧实验设置与结果是历史背景，不是本轮执行清单。

## 历史论文级实验设置

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

2026-07-13 已完成两对师生的严格配对验证。两组都使用 DAPO-Math-17K
raw 1,791,700-row pool、seed 21、200 steps、每步 4 prompt × 8 rollout，
并在 Math500/AIME24/AIME25/AMC23 上对 Step 50/100/200 做 n=8 固定
评测。主结果使用同一历史 grader 和 10,000 次 task-stratified paired
prompt bootstrap：

- Qwen3-1.7B-Base ← Qwen3-4B-Base-GRPO：Step 200 的 Block3
  Avg@8 `+0.0514`，95% CI `[+0.0353,+0.0680]`；Pass@8
  `+0.0584`，95% CI `[+0.0141,+0.1039]`，属于强单 seed 支持。
- DeepSeek-R1-Distill-Qwen-1.5B ← JustRL-DeepSeek-1.5B：Step 200
  Avg@8 `+0.0158`，95% CI `[-0.0031,+0.0356]`；Pass@8
  `-0.0015`，95% CI `[-0.0345,+0.0323]`，按预注册规则判定未复现。
- DeepSeek/JustRL 的完整 200-step 训练耗时仅相差约 0.97%，当前实现
  仍对完整 rollout 做 teacher forward；它改变的是跨 token credit
  assignment，不是减少 teacher 打分次数的计算优化。
- Block3 在 DeepSeek/JustRL 终点仍有 25.5% sign flip 和约 1.035 的
  normalized leakage。两条 run 都没有数值崩溃，因此未复现不能归因于
  OOM、NaN/Inf 或 entropy collapse。

当前总判断是：Block3 mean 具有模型对相关的有效性，但尚不是跨师生稳健的
通用 OPD 改进。完整总览和逐对审计报告见：

- `reports/block_opd_experiment_report.html`
- `reports/paired_validation/ml2/report.html`
- `reports/paired_validation/deepseek_justrl/report.html`

每个 `paired_validation` 目录中的 `report.audited.html` 是远端最终化时的
逐字节原始报告，其 SHA-256 与 `finalization_hashes.sha256` 一致；
`report.html` 使用相同 JSON 和诊断图重新生成，仅增加移动端响应式修复。

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
