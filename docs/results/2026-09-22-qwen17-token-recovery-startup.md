# Qwen3 指令版 Token OPD 恢复启动验收

时间：2026-09-22 13:34，北京时间。以下是启动快照，后续进度以服务器实时日志为准。

## 已实际启动

- ml2新队列：`runs/20260922v1_qwen17_instruct_token_recovery_seed21_ml2`。
- 控制器PID `2180368`，13:20启动；已验证PPID1、独立session。
- 正式Token训练PID `2181354`，13:23:16启动，13:34前完成第1步。
- 控制器发布：`cd2d1bde0a25844361f05a8230978eb3bd37dedb`。
- 训练和评测仍使用原冻结发布：`be736b5fac4f26ff59f4e2c21abafc456c2503f6`。
- 原始Qwen3-1.7B后训练学生、Qwen3-8B后训练教师；200步，保留50/100/150/200。

原Block3四组和初始学生一组评测已逐组校验并复用，未重跑或修改。
原失败Token目录和Ray错误日志保留，Ray日志另存新目录`failure_evidence`。

## 启动处理与对照

未修改Ray安装、超时参数、冻结训练代码或学习超参数。
新tmpfs缓存路径和CPU导入预热后，Ray真实worker门控通过：导入2.13秒，
Ray启动及worker调用13.20秒；正式训练Ray在13:23:50成功初始化。
这仅证明本次启动成功，不能据此宣称所有冷启动竞态已永久修复。

原Token的Step1保存/Step2恢复GPU测试、各rank优化器文件哈希已重新核验，
未重复GPU probe，正式训练不使用probe权重。
新旧实际`command.sh`逐字对比，仅替换输出根目录及本地缓存根目录后完全一致。

## 首步真实结果

| 指标 | Step1 |
|---|---:|
| 完整归档rollout | 32/32 |
| 长度截断 | 0/32 |
| 生成thinking标记 | 0/32 |
| 周期重复尾段 | 0/32 |
| 平均响应长度 | 1374.906 |
| 最大响应长度 | 4185 |
| 裁剪前梯度范数 | 2.593198 |
| PG loss | 0.113973 |
| Student entropy | 0.240812 |
| Teacher entropy | 0.316861 |
| Top-16 overlap | 0.727244 |
| Sign flip / leakage | 0 / 0 |

首步所有诊断数值有限，各nonfinite计数为0。
32条prompt token IDs及32条生成response token IDs均与Block3首步逐条完全相同。
这验证了首次更新前两组的实际起点；后续on-policy输出因各自更新发生变化是预期现象。
该快照只能说明启动正常，不能证明200步稳定性或最终性能。

`diagnostics/scalars.jsonl`和`diagnostics/step_00001.npz`已写入，包含绘制位置热力图
需要的数据。训练验收后自动生成诊断图，再按200、150、100、50依次完整评测。
每个权重643题×8=5144条；MATH500、AIME24、AIME25、AMC23分别计分并保留全部rollout，
使用相同历史grader，不以macro替代各benchmark结果。

## 验证与留档

- 新队列与原控制器测试：本地48项通过。
- ml2实际tokenizer参与的测试：83项通过，无跳过。
- 独立只读复核未发现启动阻断或实验配置偏移。
- 新目录证据：`startup_acceptance.json`、`startup_rollout_audit_step1.json`、
  `recovery_command_comparison.json`、`recovery_preflight.json`、`ray_gate.json`、
  `reused_resume_gate.json`、`protected_inputs.json`。

后台队列无需本地电脑保持连接。无自动重试、无checkpoint清理；若后续任务失败，
保留现场并停止，不会绕过验收继续评测或覆写旧结果。
