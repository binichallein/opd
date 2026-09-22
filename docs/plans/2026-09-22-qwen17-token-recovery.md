# Qwen3 指令版 Token OPD 恢复队列

用户授权：立即启动余下 Token OPD 训练与完整评测。只使用 ml2。

## 故障与工程处理

原队列 `20260921v1_qwen17_instruct_blockfirst_seed21_ml2` 已完成 Block3
200步及4个权重、初始学生的5组完整评测。Token独立GPU保存/恢复测试通过，
但正式训练在Ray初始化失败，尚无任何正式更新、rollout或checkpoint。

Ray2.55.1的raylet在23:59:24 UTC启动dashboard agent，23:59:39因等待
`metrics_agent_port`文件超时退出。agent首条就绪日志直到23:59:42才出现。
该症状是启动时序失败，不是CUDA OOM或训练collapse。共享文件系统冷导入是
待检验的延迟来源，不能仅凭这些时间戳认定最终根因。

官方源码 `src/ray/util/port_persistence.h` 的 `WaitForPersistedPort`
默认等待15000ms，`node_manager.cc`不覆盖该默认值，因此设置常见的
`agent_register_timeout_ms`或`raylet_start_wait_time_s`不能修复此处等待。
源码版本：https://github.com/ray-project/ray/tree/ray-2.55.1/src/ray

本次不修改已安装Ray或冻结训练代码。使用新短路径`/dev/shm/q17t`，
先预热dashboard组件导入，再实际启动CPU Ray实例并运行一个worker。
该门控通过后才启动正式训练。预热是降低冷启动风险，不保证消灭所有竞态，
若再次失败则保留现场并停止，不自动重试。原Ray失败日志先复制到持久目录。

## 不变的实验条件

- 学生：官方ModelScope `Qwen/Qwen3-1.7B`，指令/后训练版本，非Base。
- 教师：官方ModelScope `Qwen/Qwen3-8B`，指令/后训练版本，非Base。
- 学生revision：`4855588ea1a12789f2e965e5f52a9e4a24c94b2a`。
- 教师revision：`26028140be3ee69b82b1d1450179ab71bb1121b9`。
- 数据：同一DAPO训练文件，SHA256
  `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`。
- seed21；每步4题、每题8条；200步；学习率2e-6；响应上限16384。
- native chat，显式`enable_thinking=False`；训练/评测使用同一数学指令。
- actor micro-batch1、vLLM0.6；逐步保留全部rollout和观测指标。
- 在50/100/150/200保存完整模型、优化器、调度器、数据及各rank RNG状态。
- 训练/评测代码仍使用冻结`be736b5fac4f26ff59f4e2c21abafc456c2503f6`。
- 重新从原始学生开始，`resume_mode=disable`，不加载Block3或probe权重。

## 自动顺序与验收

新队列：`20260922v1_qwen17_instruct_token_recovery_seed21_ml2`。

1. 锁定新旧队列，确认原进程退出，故障发生在首次更新前。
2. 校验冻结代码、受保护模型/数据/原run card以及五组旧评测和raw归档。
3. 校验原Token保存/恢复验收、优化器文件哈希和probe轨迹；不重复GPU测试。
4. 生成Token命令，与原Token配置比较，仅允许输出位置变化。
5. CPU Ray预热/真实worker门控通过后，正式训练200步。
6. 验证全部checkpoint、6400条rollout及与Block3相同的数据/请求seed，画诊断图。
7. 按200、150、100、50完整评测，每组643题×8=5144条，保留所有输出。
8. MATH500、AIME24、AIME25、AMC23分别报告Avg@8、Pass@8、截断和格式统计；
   使用历史grader。结合旧5组结果输出9组对照，禁止合并成单个macro主结论。

原目录、失败日志、已有评测不覆写；无checkpoint清理、无自动重试。
CPU门控成功仅说明启动链路当次可用，最终启动必须以正式训练实际step日志为准。
