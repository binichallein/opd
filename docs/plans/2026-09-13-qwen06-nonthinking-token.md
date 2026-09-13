# Qwen3-0.6B 非 Thinking Token OPD

用户已明确批准直接启动新的 0.6B Token OPD，要求 GPU 验证 thinking 关闭、
训推 prompt 一致、完整训练 rollout 保存和原有过程监控。仅使用 ml2，
不恢复旧队列，不启动 Block3，不改写旧运行。

## 固定协议

- 官方 Qwen3-0.6B-Base，revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd；
  现有公共 Qwen3-4B-Base-GRPO 教师。均沿用已校验本地模型，不加入 SFT。
- 原 DAPO pool，1791700行，SHA256
  cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf。
- seed21，200步，4 prompts/step，每题名义8条，学习率2e-6，PPO epochs1，
  actor micro-batch1，vLLM占比0.6，四张A100。temperature1、top_p0.9、
  prompt上限2048、响应上限16384，保留旧配置的其他训练超参数。
- prompt使用现有评测的数学指令生成规则，共用代码；不含 Math problem:
  包装或强制think指令，显式enable_thinking=False。保留Qwen模板表达关闭
  thinking的空、已闭合think块。普通逐步解答仍然允许。
- 停止集合与评测一致：[151643,151645]。vLLM真实生成长度用于response mask，
  避免把停止后的padding算成响应；此行为单独标记为新协议，不混入旧对照。
- Step50/100/200保留模型、优化器、scheduler、各rank RNG、dataloader状态，
  不自动删checkpoint。后续完整评测按MATH500/AIME24/AIME25/AMC23分开计分；
  本次启动范围仅Token训练，不重启已暂停的旧完整评测。

## 实现与验收顺序

1. [x] 单元测试：共用prompt规则、显式false、停止后mask、无损记录、拒绝覆盖。
   全库524 passed、2 skipped；远端独占目录和rename发布验证通过。
2. [ ] 独立GPU推理：固定16题、原始token/文本/停止原因；验证真实输入为关闭
   模式，统计新生成的think标签。发现违规先报告，不用静默删标签伪装修复。
3. [ ] 新运行下的两步恢复门：Step1保存，重启从状态恢复到Step2。检查所有rank
   状态、全部32条/step原始rollout、prompt匹配及非有限值。
4. [ ] 从Base独立开始正式200步，绝不从probe或旧Token权重继续。
5. [ ] 启动后核验首批轨迹、梯度、停止原因和诊断热图，继续保留正式日志。

## 观测与数据安全

- 每步更新前归档全部实际生成响应，含数据行标识、group/traj ID、prompt及
  response token IDs、保留特殊token的文本、真实finish/stop reason、采样参数、
  EOS、mask、rollout logprobs，以及run/attempt/step/source/protocol身份。
- 使用按attempt/step分开的独占创建压缩JSONL，完整写入后原子发布，拒绝覆盖。
  不消耗采样RNG，不改变loss，不把重新采样当历史训练轨迹。
- 每步训练日志：PG loss、裁剪前grad norm、student entropy、长度、截断率；
  原有诊断节奏Step1及每5步：student/teacher entropy、signed/absolute gap、
  top16 overlap、两侧overlap mass、overlap-token advantage、sign flip、
  weighted sign flip、leakage magnitude及归一化版本、更新后ratio诊断。
- 位置统计128token/bin，stride1，保留npz并生成热图。Token k=1的block信用
  指标理论上退化为0，作为实现一致性检查而非新方法收益。
- 新增真实finish reason截断率、生成think标签率。仅检测响应中的标签，不能
  把prompt里的空think控制块误报为thinking开启。
- GPU测试不能证明Base永远遵从控制；监控持续保留违规率，不声称解码强约束。

## 启动尝试

- v2 / 8a0a2d3：GPU检查在CPU测试输入构造处报缺少batch字段，未生成响应、
  未启动训练；源版本与失败目录均保留。修正为真实DataProto输入后以v3启动，
  没有原地覆盖失败记录或冻结版本。
