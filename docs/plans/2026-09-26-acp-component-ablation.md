# ACP：共享反馈与损失倍率消融

## 授权与研究问题

用户在组件消融解释后授权在ACP启动。只新增adv3和scale3，不启动ml2协议实验、
不重训已完成的Token/Block3，不新增师生或seed。Qwen3-8B教师、Qwen3-1.7B学生均为
固定ModelScope官方指令版权重。原始学生不重复评测。

- adv3：平均相邻三个有效token的advantage，逐token PPO ratio，逐token reduction。
- scale3：保留每个token自己的advantage和PPO ratio，将策略损失乘所在块的有效token数n_j。
  完整块乘3，尾块按实际数量处理；不修改学习率、熵项或KL项，不把它称为Adam更新三倍。
- 对照：ACP原始100步32x1 Token及完整历史Block3，运行代码1867093。
  原Block3还使用联合ratio，两新增组不构成完整三因素析因实验。

## 固定协议

两组各100步、32题x1回答、seed21、legacy request seed、lr2e-6、PPO minibatch32/epoch1；
原生Qwen chat、显式enable_thinking=False，训练评测一致。响应16K，micro1，vLLM0.6。
保存25/50/75/100全部完整状态，不删除；每步保存全部rollout，Step1及每5步诊断和位置热图。
每臂先保存1步、退出、恢复到2步验收，probe使用100步scheduler。正式训练从原始学生重启。
顺序为adv3训练→100/75/50/25完整评测→scale3训练→100/75/50/25完整评测。
四benchmark各自报告Avg@8、Pass@8、格式错误、截断，不合并分数。沿用历史grader，每题8条。

## 正确性与可恢复性

CPU测试覆盖loss、梯度、异号优势、尾块、mask及stop-gradient；旧方法回归不变。
新增实现及诊断变化使用旧/新文件hash白名单，其余源文件、模型/数据/评分器锁定。
两组与已完成对照的配置、100步题目计划、每步实际输入逐一审计。
adv3实际ratio粒度为1；scale3无信用共享，符号翻转/重分配诊断用粒度1。
scale3的信用重分配零值只反映没有共享，不代表任务信用更正确。
任何失败停止队列，保留日志，不自动调参或重跑。

所有持久产物在AFS。使用服务端nohup启动，不触碰sshd；SSH断线不影响后台任务，
但平台生命周期仍可能终止任务。保留完整状态以便核验后恢复。

## 进度

- [x] ACP四卡空闲与原对照8项完整评测核验。
- [x] 新loss和控制器测试先失败，进入实现。
- [x] CPU回归、不可变部署和ACP实际环境验收。
- [x] adv3 GPU保存1步、恢复到2步门禁。
- [x] adv3正式训练首批更新与轨迹验收。
- [ ] 两组全部训练、8项完整评测、归档。

## v2启动与审查记录（已停止）

- 冻结代码 `84d95e408a493389bdee5cc52e76717a722cfcf1`，已推送GitHub。
- ACP实际环境215项测试通过，包括固定真实tokenizer；一项依赖完整Git历史的测试
  只在本地运行并通过，AFS不可变部署不携带Git数据库。
- 24组与1867093的Token/完整Block3数值回归：loss、辅助指标和梯度逐值完全一致。
- 模型、数据、完整pip freeze、Python版本与旧运行一致；3200题目位置计划一致。
  实际collector核对643道评测题和64道验收题，训练评测prompt token一致，thinking关闭。
- 控制器PID153434于2026-09-26 14:45北京时间以nohup/独立session启动。
  此时只代表自动队列已启动，尚未证明GPU保存恢复门禁或正式训练已完成。
- 运行根目录 `/mnt/afs/202609/tyf-qwen-opd/runs/20260926v2_qwen8_to17_instruct_components_n1_seed21_acp`。
  CPU证据在 `preparation/`，动态进度读 `queue_state.json` 与各子任务日志。
- 独立复核发现逐token ratio的诊断值被按3-token块展开，会在首步诊断报维度错误。
  于student GPU smoke阶段主动停止控制器153434，未进入训练，也未产生训练权重。
  新增执行真实trainer展开代码的12项测试，先复现6项失败，再修复展开宽度。
- v3运行根目录为 `runs/20260926v3_qwen8_to17_instruct_components_n1_seed21_acp`。
  保留v2现场，不覆盖旧部署。算法、数据和训练超参数不变，只修复诊断与重建独立队列。

## v3启动记录

- 冻结代码 `ba83e69025fcc0f2e533416328f041dd23304afb`，ACP测试227项通过。
- adv3保存与恢复两个进程均exit0，四个rank的训练状态、64条轨迹及位置诊断通过验收；
  恢复前的Step1文件hash未改变。检查不保证vLLM逐位重放。
- 两批探针均截断0/32、生成think标签0/32、周期重复尾段0/32，梯度范数约0.760/0.870。
- 正式adv3进程PID176726于15:40:50北京时间启动，独立从原始学生权重开始，
  控制器PID155322。此条仅记录进程启动，首批正式更新需另行验证。
- 15:53正式step8已确认，前192条输入与两组对照一致，无截断、think标签或周期重复尾段；
  Step1/5标量及位置诊断通过。后续训练、评测与scale3尚未完成。
- 所有小型JSON证据保存在 `results/qwen17_components_acp_20260926/v3_*.json`；
  checkpoint、完整原始轨迹与完整日志保留AFS，不上传Git。
