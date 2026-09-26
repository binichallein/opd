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
- [ ] CPU回归、不可变部署和ACP实际环境验收。
- [ ] adv3 GPU保存恢复门禁与正式训练启动。
- [ ] 两组全部训练、8项完整评测、归档。
