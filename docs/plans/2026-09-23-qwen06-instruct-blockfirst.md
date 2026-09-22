# Qwen3-4B到0.6B指令版：配对训练启动方案

## 用户授权与范围

用户选择官方指令版4B教师、0.6B学生，确认复用8B到1.7B指令版的训推一致prompt后授权启动。
沿用该轮完整流程：先Block3 Mean训练及完整评测，再Token OPD训练及完整评测。
模型取自本次已通过的ModelScope资产，分别为`Qwen/Qwen3-4B`、`Qwen/Qwen3-0.6B`。
两组都从同一个原始学生独立初始化，不从教师诊断、probe或另一训练组的权重继续训练。

## 不变的对照条件

- ml2四张A100 80GB；不访问train，不使用私有模型。
- 原DAPO派生训练文件，SHA256 `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`，1791700行，seed21。
- 每步4题、每题8条，200步，学习率2e-6，最大响应16384 token；沿用独立请求seed规则。
- actor/ref microbatch1、rollout logprob microbatch4、vLLM memory utilization 0.6。
- `qwen3_native_chat_no_thinking_boxed_v1`，显式false、原生chat和空闭合think预填充，停止集合[151645,151643]。
- 保留历史Block3 Mean的共享advantage、联合block ratio、block损失归一化；Token目标不改。
  不把本轮称为只修改advantage平滑的纯消融。
- 每组分别完成真实GPU Step1保存、退出、恢复Step2检查，随后从原始学生重新启动正式200步。
- 每组保留Step50/100/150/200完整模型、优化器、调度器、RNG、dataloader恢复状态，不删除任何checkpoint。
- 每步保存全部32条rollout和既有诊断指标，包括师生熵与gap、top16 overlap及mass、overlap advantage、
  PG loss、裁剪前梯度范数、长度/截断与block专属指标；训练完成自动画位置热图。
- 每个权重分别完整评测MATH500/AIME24/AIME25/AMC23，每题8次，seed21到28、历史grader、16K预算不改。
  每benchmark独立报告Avg@8、Pass@8、格式与截断，保留全部原始评测输出，不计算macro。

## 队列与验收

新运行目录：`runs/20260923v2_qwen06_instruct_blockfirst_seed21_ml2`。
新短tmpfs缓存：`/dev/shm/q06j`；新不可变部署，不覆盖旧运行。
顺序为Block3恢复门禁 -> Block3 200步 -> 四权重/轨迹/指标验收与热图 ->
Block3 Step200/150/100/50评测 -> 原始学生评测 -> Token恢复门禁 -> Token 200步 ->
验收/热图 -> Token Step200/150/100/50评测 -> 分benchmark配对汇总。

教师门禁引用已封存八组诊断中的`i4_i06`，汇总SHA256为
`e19bad52b2fa48c9b074c0fdc661689287b5d35fdc49ecd9e967a6acdd62988c`。
重新验证模型revision、资产和证据哈希；实际训练collector/eval/教师prompt逐token一致。
每臂训练前添加既有CPU Ray导入预热和真实worker启动检查，防止此前遇到的冷启动问题；
不修改Ray超时或训练参数。失败停止并保留现场，不自动改参重试。

推理诊断通过不是OPD涨点保证；本轮是新师生对的受控训练验证，不能与早期0.6B Base旧协议混合计分。

## 启动兼容性修复

v1于03:22:53启动，03:24在生成训练命令前停止，未进行GPU训练。
错误是推理资产准备器未生成旧训练启动器所需的`SOURCE_REVISION`元数据文件。
保留v1的全部目录、日志和缓存；新增v2目录及缓存，从已核验的固定ModelScope revision派生该标记，
不修改模型、tokenizer或训练参数。标记已有但不匹配时拒绝覆盖，标记哈希加入保护清单。
修复后增加真实prepare-only命令检查，再启动GPU队列。

## 执行进度

- [x] 用户授权和模型选择；ml2空闲，资产与磁盘空间就绪。
- [x] 复用原控制器、新增独立配对入口和门禁测试；66项本地CPU测试通过。涉及torch的协议回归转到ml2环境验证，本地conda的torch存在iJIT动态库导入错误。
- [x] CPU回归、配置对齐检查、提交与不可变部署：86a4122；ml2上370项通过，Git历史对照单测本地通过。两臂真实prepare-only和全部输入token ID核验通过。
- [x] 服务端nohup启动，PPID/SID确认及CPU Ray真实worker门禁通过；启动记录已备份本地。
- [x] Block3 GPU Step1更新、32条轨迹保存、完整checkpoint及四rank优化器审计通过；0截断、0生成think标签、0周期重复尾部。
- [x] Block3从Step1恢复到Step2的完整门禁通过；两步共64条轨迹均无截断、生成think标签或周期重复尾部，证据已本地备份。
- [x] 正式Block3于03:57:07独立启动，PID2743663；日志确认200步、禁用resume，未复用probe权重。
- [ ] 正式Block3首步rollout/监控验收。
- [ ] 自动完成两臂训练、九次完整评测及最终归档。
