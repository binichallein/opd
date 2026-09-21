# Qwen1.7 Instruct Block-First Implementation Plan

**Goal:** 按最新授权，先Block3 Mean训练200步及完整评测，再从同一原始学生启动Token OPD训练200步及完整评测，全部自动串行。

**Architecture:** 新的不可变训练部署和独立控制队列，沿用Revisiting OPD历史损失、数据与监控实现。
只新增已验收指令提示的协议适配和本次编排/审计；旧部署、旧权重、旧结果不改写。
正式训练前分别做Step1保存、退出、恢复Step2的真实GPU门禁。失败停队列，保留全部现场，不自动调参或重试。

**Tech Stack:** 既有verl/FSDP、vLLM0.11.0、torch2.8.0、Transformers4.57.6、pytest、ModelScope固定资产。

## 用户已确定的设计

- ml2唯一机器，四张A10080GB；教师官方 `Qwen/Qwen3-8B`，学生官方 `Qwen/Qwen3-1.7B`。
  复用已通过验收的ModelScope完整revision及原始资产，不用Base或历史训练checkpoint。
- 新根目录 `runs/20260921v1_qwen17_instruct_blockfirst_seed21_ml2`，独立短缓存 `/dev/shm/q17i`。
- 阶段顺序：Block3 Mean恢复门禁 -> Block3 Mean200 -> 检查四权重/全部轨迹/生成监控图 ->
  Block3评测200/150/100/50 -> 原始学生完整评测 -> Token恢复门禁 -> Token200 ->
  检查四权重/全部轨迹/配对输入/生成监控图 -> Token评测200/150/100/50 -> 分benchmark比较。
- 每臂独立从原始学生初始化；正式训练不接着probe或另一臂续训。
- 四个checkpoint为50/100/150/200，保留模型、四rank优化器、调度器、RNG和dataloader状态；不删除或压缩掉中间状态。
- 沿用同一1791700行DAPO训练文件及hash、seed21、每步4题x8条、lr2e-6、200步、16K响应预算。
  每条rollout独立确定性seed，按题目/sample/step派生；两组对齐输入，而不要求生成输出相同。
- actor microbatch1、teacher/ref logprob microbatch1、rollout logprob microbatch4、vLLM占比0.6；不暗改学习率或loss缩放。
- 保留历史Block3联合ratio与block reduction，Token使用历史sampled-token目标；不声称纯advantage平滑消融。
- 新协议 `qwen3_native_chat_no_thinking_boxed_v1`：题目strip后追加
  `Please solve the problem step by step and put the final answer in \\boxed{}.`，使用原生chat及显式enable_thinking=False。
  精确匹配已完成教师验收的提示；训练和完整评测均使用同一函数与token ID核验。
- 使用真实EOS151645及停止集合[151645,151643]，有效长度mask处理多停止token；不修改模型tokenizer。
  此mask与旧Base历史EOS mask的区别必须记录，两条新训练臂采用完全相同规则。
- 每步保存32条lossless rollout；每步保存学生/教师熵、entropy gap、top16 overlap/mass、
  overlap advantage、PG loss、裁剪前grad norm、长度/截断及block专属指标，自动画位置热图。
- 完整评测沿用MATH500(500)、AIME24(30)、AIME25(30)、AMC23(83)，每题8次、历史grader及历史采样。
  共9个模型评测，每个5144条；每benchmark单独报告Avg@8、Pass@8、格式和截断指标，保存全部原始输出。
- 本轮教师能力门槛已通过，但续写截断6/64、周期重复3/64接近阈值；不能据此保证OPD稳定或Block3有效。

## 实现任务

1. 新协议适配及回归测试：`opd_ext/math_protocol.py`、既有训练/eval入口与最小vendor适配，更新patch和manifest。
   旧协议输出不变；新协议须与资格验收prompt逐token一致；测试显式thinking、EOS、mask及归档。
2. 新 `scripts/run_qwen17_instruct_pair.py` 及测试：显式模型/版本/数据校验、配对run card、probe/恢复/完整checkpoint审计、
   自动Block3优先顺序、完整eval验收和分项汇总；不修改旧控制器的全局常量。
3. 验证实际训练collector/eval token一致、请求seed和数据顺序一致；两种方法GPU更新与恢复均须独立验收。
4. CPU回归及独立代码复核通过后提交并部署新的immutable runtime；启动一次服务端nohup队列。
5. 验证控制器PPID/SID、首probe进展、错误日志和后续自动阶段，记录实际启动证据。

## 进度

- [x] 最新训练顺序、200步和四权重策略确认；旧验收结束，ml2 GPU空闲。
- [ ] 新提示协议及测试。
- [ ] 自动训练/评测控制器及测试。
- [ ] 独立复核、实际环境回归和不可变部署。
- [ ] 服务端队列启动、Block3恢复门禁和正式训练启动确认。
- [ ] Block3完整评测、原始学生评测、Token训练和完整评测。
