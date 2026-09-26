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
