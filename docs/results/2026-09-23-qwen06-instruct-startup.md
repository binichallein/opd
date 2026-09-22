# Qwen3-4B到0.6B指令版：启动记录

## 当前运行

- ml2运行目录：`runs/20260923v2_qwen06_instruct_blockfirst_seed21_ml2`。
- 根目录：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
- 冻结部署：`deployments/86a4122df91c5d1b76091619ad13048335d3b949`。
- 2026-09-23 03:32:48北京时间启动服务端nohup控制器，PID `2722747`；已核验PPID为1，SID同PID。
- [完整训练与评测方案](../plans/2026-09-23-qwen06-instruct-blockfirst.md)。

启动时已完成本地67项CPU回归、ml2上370项回归，以及依赖Git历史的1项本地对照测试。
真实0.6B tokenizer的协议测试全部通过；没有以缺tokenizer的跳过测试替代实际检查。
两组真实prepare-only命令均成功，生成的run card通过逐项配对审计。

200步、seed21、4题x8条、lr2e-6、16K响应预算、microbatch1和vLLM0.6保持上一对指令模型设置。
先Block3 Mean训练及四权重完整评测，再原始学生评测，再Token训练及四权重完整评测。
每组正式训练独立从原始学生初始化；Step50/100/150/200的完整恢复状态全部保留。

## Prompt验收

64道教师诊断题，以及MATH500/AIME24/AIME25/AMC23共643道完整评测题，
已使用真实训练collector、评测函数和教师tokenizer逐token比对。
协议为`qwen3_native_chat_no_thinking_boxed_v1`，显式关闭thinking，停止集合[151645,151643]。
全部评测题的输入身份哈希为
`6d71d295bdad757cff7ee1eb43bd6295448a64ca0e3e7b5f39d202fe0fb75fbd`，
与上一轮官方8B到1.7B指令版实际保存的`input_contract.json`完全一致。

模型为官方ModelScope `Qwen/Qwen3-0.6B`（学生，revision `09b42cad3d112e832108974449ccb5e8e0f5b5d1`）
和`Qwen/Qwen3-4B`（教师，revision `2c54d5a09e7e92d4f5126b92a5a457448c9593e6`）。
未使用Base或旧训练权重。教师门禁引用[八组筛选](2026-09-23-qwen06-teacher-screen-final.md)的`i4_i06`。

## 保留的失败现场

v1运行目录`runs/20260923v1_qwen06_instruct_blockfirst_seed21_ml2`在03:24停止，未发生GPU训练，
没有checkpoint或训练rollout。原因是新推理资产准备器没有生成旧训练启动器必需的`SOURCE_REVISION`。
v2仅从已核验的ModelScope快照revision生成缺失标记，保留所有权重、tokenizer和旧目录不变；
标记冲突或为符号链接时拒绝覆盖，新增标记哈希纳入保护清单。
代码、测试与修复记录均已提交，未删除失败记录或把失败当作成功训练。

## 自动验收阶段

03:35，CPU Ray导入预热与真实worker门禁通过；03:35:50启动Block3的Step1小步保存测试。
只有在Step1保存、退出、恢复Step2、四rank优化器/RNG/dataloader及rollout审计都通过后，
才会独立启动正式200步。启动小步任务不等于已经完成正式更新。

每步保留32条完整轨迹和熵、overlap、梯度、长度、截断、sign-flip及leakage指标；
训练完成后自动生成位置热图，完整评测逐benchmark输出Avg@8、Pass@8、格式和截断。
队列失败时停下并保留现场，不自动调参重试。

动态状态以`queue_state.json`、`queue.log`、`queue_jobs/*/logs/job.log`为准；
本记录的启动状态不替代后续训练/评测完成验收。

### Step1实测

03:46，Block3小步训练与checkpoint审计均以exit code 0结束。
四rank优化器状态检查通过，保存了Step1完整checkpoint；恢复Step2于03:47:19启动。
此处仅记录启动健康性，不是正式200步结果，也不证明Block3相对Token有收益。

| Step1指标 | 实测值 |
| --- | --- |
| 完整轨迹 | 32条 |
| 截断 / 生成think标签 / 周期重复尾部 | 均为0/32 |
| 平均 / 最大响应长度 | 778.31 / 1655 token |
| 裁剪前梯度范数 | 11.0013 |
| 学生 / 教师熵 | 0.4064 / 0.3539 |
| Top16 overlap | 0.6977 |
| Sign flip / 加权sign flip | 0.2454 / 0.0639 |
| 归一化leakage | 1.0360 |
| 熵、advantage和更新后block ratio的非有限计数 | 全部为0 |

进程退出时日志含被忽略的DataLoader清理异常，发生在保存完成之后；
训练和审计退出码均为0。保留原日志，不将这条清理警告抹去或当作GPU训练失败。

[Step1验收、逐项指标及来源哈希](../../results/qwen06_instruct_20260923/probe1/acceptance.json)
已归档。本地`/home/tyf/paper/outputs/qwen06_instruct_20260923/probe1_0346`
保存10个原始文件，包括32条压缩轨迹、位置诊断和完整运行日志；10个文件SHA256均重新核验。
完整模型与优化器文件仍保留在ml2；本地这份记录不是完整权重备份。

### 恢复验收与正式启动

03:56，恢复任务完成Step2；03:57:07，控制器通过全部恢复检查并自动启动正式Block3 Mean。
正式进程PID为`2743663`，启动日志确认`total_training_steps=200`、`resume_mode=disable`，
从原始学生重新开始。03:59检查时仍处于数据与worker初始化，没有正式训练step记录。

Step2保存、退出及审计的退出码均为0。恢复门禁检查Step1和Step2的四rank状态、
调度器步数、RNG、优化器和dataloader位置；该验收不承诺vLLM逐bit复现。
两步共64条轨迹，截断、生成think标签、周期重复尾部均为0。
Step2平均长度1445.41、最大3273 token，裁剪前梯度范数6.3069，
学生熵0.4680、sign flip 0.2513，熵/advantage/block ratio的非有限数值计数仍全部为0。

[恢复验收与正式启动快照](../../results/qwen06_instruct_20260923/resume/acceptance.json)
记录全部门禁证据、两步标量和14项文件哈希。
原始文件保存于`/home/tyf/paper/outputs/qwen06_instruct_20260923/resume_0357`，
包括Step2的32条原始轨迹、位置诊断和恢复日志；14项哈希全部核验。
Step1快照仍单独保留，没有覆盖或混入正式训练结果。

### 正式更新验收

04:07确认正式训练完成Step3，04:08:48完成Step5/200。
前五步均各保存32条轨迹，共160条，标量记录的截断率及生成think标签率均为0；
裁剪前梯度范数为10.9974、6.2300、6.7468、5.6937、6.7063，未检出非有限诊断数值。
这仅说明启动阶段未观察到上述异常，不是最终稳定性或准确率结论。

另外独立审计了正式Step1至3的96条完整轨迹：模型实际输入、历史DAPO题目顺序、
逐请求seed、停止token、mask、有效长度及logprob都通过既有审计函数；
截断、生成think标签和周期重复尾部均为0。
该审计与小步恢复测试分开，未将probe轨迹当作正式训练结果。

[正式首三步验收](../../results/qwen06_instruct_20260923/formal_steps1_3/acceptance.json)
保存逐步标量及原始文件哈希。本地快照
`/home/tyf/paper/outputs/qwen06_instruct_20260923/formal_steps1_3`
包含三步的原始压缩轨迹、哈希侧文件和位置诊断，9个原始文件已逐一核验。
后续训练仍在ml2运行，尚未完成四权重完整评测或Token对照。

## 启动归档

[启动配置、输入核验和来源哈希](../../results/qwen06_instruct_20260923/startup/startup.json)
保存v2队列、两臂run card、Ray门禁、CPU预检以及v1失败状态。
本地原始启动快照位于`/home/tyf/paper/outputs/qwen06_instruct_20260923/startup_0332`，
共22个文件，包含失败日志。它不包含尚未完成的正式训练轨迹或checkpoint，不能称为最终训练备份。
