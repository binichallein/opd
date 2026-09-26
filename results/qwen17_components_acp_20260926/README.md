# ACP 8B→1.7B 指令版组件消融

本目录记录新增adv3与scale3的验收和后续结果，不覆盖已完成的Token/Block3。
训练方案见 `docs/plans/2026-09-26-acp-component-ablation.md`。

- adv3：只共享相邻3-token平均advantage，逐token ratio和归一化。
- scale3：不共享advantage，逐token策略损失乘块内有效token数，不改学习率。
- 两组各100步、32x1、seed21；25/50/75/100完整状态全保留，每个权重完整n8评测。
- 原始权重、训练与评测rollout保存在AFS，Git只保存小型配置、证据摘要和图表。
- v2固定代码84d95e4；控制器153434于14:45启动，14:49在student GPU smoke阶段主动停止。
  独立审查发现位置诊断ratio宽度不匹配，尚未进入训练，未产生训练权重。
  修复后改用独立v3目录和新部署，保留v2现场；CPU验收不等于正式训练成绩。

## 已完成的启动前检查

215项ACP测试、24组旧loss逐值回归、旧运行环境一致性、100步题目计划一致性、
实际collector输入与非thinking模板一致性均通过。
本地默认Python存在Torch符号冲突，另一个本地环境存在tokenizers版本冲突；
未更改这些环境，也未更改ACP训练依赖。相关真实tokenizer检查已在ACP通过。

新测试执行真实trainer的位置展开代码，覆盖4种模式和3种长度；先复现6项错误，
修复后相关70项测试通过。v3的GPU保存恢复、正式训练和完整评测状态将随验收更新。

## v3已核验进展

- 冻结运行代码 `ba83e69025fcc0f2e533416328f041dd23304afb`，控制器PID155322。
- ACP上227项测试通过，1项仅依赖Git历史的测试在本地通过；不改服务器依赖。
- 师生实际GPU smoke各4条均通过，新增think标签均为0。
- 15:31北京时间：adv3保存探针正常退出（exit0），Step1四rank完整状态和32条轨迹审计通过；
  截断0/32、生成think标签0/32、周期重复尾段0/32，梯度范数约0.760，非有限诊断为0。
  这些是探针结果，不是正式训练或benchmark成绩。恢复进程已启动，正式更新尚未开始。
- 保存探针退出时出现DataLoader worker清理警告；旧对照探针也有此警告，
  当前退出码为0且cgroup的oom/oom_kill计数为0。后续恢复验收仍必须通过。
- v3配置证据单独存为 `v3_queue_manifest.json` 和 `v3_baseline_alignment.json`，
  不覆盖v2的启动前检查记录。AFS运行目录为
  `/mnt/afs/202609/tyf-qwen-opd/runs/20260926v3_qwen8_to17_instruct_components_n1_seed21_acp`。
- 15:53保存恢复与正式启动验收均已通过，正式进度8/100。前192条输入匹配两组旧对照，
  截断/生成think标签/周期重复尾段均为0；Step1/5标量和位置诊断通过。
  证据为 `v3_startup_acceptance.json` 及 `v3_adv3_*gate.json`；尚无新benchmark成绩。
- 位置图沿用历史字段名 `post_update_block_*`，但adv3/scale3的实际ratio宽度为1。
  不将这些ratio数值直接与历史Block3的宽度3指标横比；熵、重合度、长度等指标口径不变。
  scale3的共享信用指标为零，不表示没有损失缩放，也不证明其信用分配更正确。
