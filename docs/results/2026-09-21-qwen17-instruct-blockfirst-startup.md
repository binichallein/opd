# Qwen1.7 指令版 Block3优先队列启动

## 授权与顺序

用户指定先Block3训练，再完整评测，然后Token OPD训练与评测，自动串行。
教师官方ModelScope `Qwen/Qwen3-8B`，学生 `Qwen/Qwen3-1.7B`，均为已验收指令版。
两臂独立从同一个原始学生初始化，200步，保存全部50/100/150/200完整训练状态。

固定顺序：Block3两步恢复门禁、正式训练、四权重验收/监控图、评测200/150/100/50、
初始学生完整评测、Token两步恢复门禁、正式训练、四权重验收/监控图、评测200/150/100/50。
每次完整评测均为643题、每题8次，MATH500/AIME24/AIME25/AMC23独立计分，历史grader。

## 冻结配置

- 数据与历史相同：1791700行DAPO文件，SHA256 cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf。
- seed21，每步4题x8条，lr2e-6，响应上限16384，temperature1、top_p0.9。
- actor/ref microbatch1、rollout logprob microbatch4、vLLM显存比例0.6。
- 新协议 `qwen3_native_chat_no_thinking_boxed_v1`，原生chat显式关闭thinking，保留空闭合控制前缀。
- 训练与评测共用相同题目追加指令和原生token IDs；EOS151645，停止集合[151645,151643]，有效长度mask。
- 历史Block3的共享advantage、联合PPO ratio和block归一化均保留，不是纯advantage平滑消融。
- 每步保存全部32条原始rollout与诊断，训练后自动画标量曲线和位置热图。
- 全部checkpoint、原始输出、失败现场保留；异常停队列，无自动调参重跑或删除。

## 启动证据

- 北京时间2026-09-21 23:38，ml2后台nohup/setsid启动。
- 控制器PID1896918，启动后确认PPID1、SID1896918。
- 新根目录：`runs/20260921v1_qwen17_instruct_blockfirst_seed21_ml2`。
- 固定部署：`deployments/be736b5fac4f26ff59f4e2c21abafc456c2503f6`。
- 控制日志：根目录`controller.log`，实时状态`queue_state.json`。
- 开始时四GPU均空闲；首次子任务是runtime_hashes，尚不能称正式训练已开始。

## 工程验证

- 实际ml2环境438项CPU回归通过，另一个依赖Git历史的测试在本地通过。
- 本地真实tokenizer测试最初受transformers/tokenizers版本冲突影响；ml2实际环境中的对应测试全部通过。
- 真实训练collector检查全部643道评测题，师生tokenizer与评测输入IDs一致。
- 64道已完成教师验收题的输入与新训练协议一致，输入汇总SHA256为
  `6d71d295bdad757cff7ee1eb43bd6295448a64ca0e3e7b5f39d202fe0fb75fbd`。
- 54项模型/数据/资格证据预检通过，历史loss与diagnostics文件逐文件hash一致。
- 补充了主机身份与历史benchmark固定hash校验；独立复核无剩余阻断问题。
- GPU实际更新、完整状态保存、退出恢复必须由运行中的两步门禁另外验收，不以CPU测试代替。

上述是启动快照，不是训练完成或方法有效的结论。后续以实时状态及验收文件为准。
