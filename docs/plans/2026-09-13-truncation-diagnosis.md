# 0.6B 高截断率诊断

用户于 2026-09-13 要求停止当前完整评测，优先查明 0.6B 高截断率原因，
与历史 1.7B 对比。仅使用 ml2，不恢复训练或 benchmark 队列。

## 状态

- 14:04 北京时间已停止 Qwen06 队列控制器及 Step50 n=8 评测。
  控制器退出后的 `error=143`、评测 `exit=-15` 是主动中止。
- 已复核四个旧 vLLM engine 和评测 PID 均退出，四卡显存均降至 4 MiB。
- Token 训练已完成，Step50/100/200 保留；Block3 未启动。
- 当前只进行额外的诊断推理，不更新权重，不把它当完整 benchmark 成绩。

## 同题对照

1. 使用两组历史训练相同的 DAPO 数据文件，沿原 SequentialTaskSampler
   的 seed21 顺序取前 32 个 prompt，即最初 8 个训练 batch 的题目。
2. 复用环境的原始 `MATH_TEMPLATE` 和聊天模板。该环境模板明确要求在
   `<think>...</think>` 内推理，不能只使用裸题目或评测模板。
3. 重新构建的 Step1、Step5 prompt 多重集哈希与两组训练记录均完全一致。
   cohort SHA256 为 `dd9e79bc95378ae3ac4cbb194b511f3fed7d1b6fb9050ac29ee8d93ea3f33aee`。
4. 四个模型状态：官方 0.6B Base、官方 1.7B Base、0.6B Token Step50、
   1.7B Token Step50。每个状态同题生成一次，seed21、temperature1、
   top-p0.9、max_tokens16384、只按模型 EOS151643 停止。
5. 四个状态使用相同的诊断引擎配置：vLLM0.11、BF16、TP1、eager、
   memory0.6、max_num_seqs32。不是原 FSDP/vLLM 训练进程的逐位复现；
   原训练每题名义8条，本诊断每题1条，不能冒充原始 rollout。

## 证据与假设

- 保存每条实际生成的 prompt IDs、完整 response IDs、保留特殊 token 的
  原文、finish_reason、stop_reason、长度、seed、源文件哈希。
- 完成一条立即写入并 fsync，不等待全部样本完成。独立文件拒绝覆盖。
- 分开报告引擎的 `finish_reason=length` 与长度恰好触顶，避免把 EOS 恰好
  出现在末位的样本错误归类。
- 检查 EOS151643、消息结束符151645 的第一次出现位置。如果触顶回答
  早已包含151645，那么停止协议是可直接验证的贡献因素。
- 检查最后2048 token 的周期重复：至少256 token，周期1-128，至少4次
  重复、相隔一个周期的位置一致率>=99%。同时保留4-gram重复比例与原文，
  不用一个启发式指标代替人工检查。
- Base 对比用于判断是否训练前就发生；Step50 对比用于判断是否训练后
  保留/加重/缓解。32题是小样本机制诊断，不推断总体发生率。
- 如发现停止符或强制思考模板的线索，再在相同题目上做单因素推理干预；
  不改任何原始训练配置、不启动新训练、不恢复完整评测。

## 实现与产物

代码：`scripts/diagnose_token_truncation.py`。
初始6个单元测试检查长度边界、停止信息保真、周期检测和禁止覆盖；
v3 增加 no_think 单因素干预测试，总计7个。

独立部署：
`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/analysis_releases/20260913_truncation_probe_v2`。

诊断产物：
`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/analyses/20260913_qwen06_truncation_probe_v2`。

准备阶段 v1 漏掉环境的 MATH_TEMPLATE，被 prompt 哈希检查拒绝，未运行
任何 GPU 诊断。修正后使用独立 v2 代码，未覆盖 v1 或任何冻结训练部署。

## 提示格式消融

第一轮 128 条生成已全部完成，四个任务均 exit0。所有输出均未出现
消息结束 token151645，因此本轮不额外运行双 EOS 停止消融。

第二轮使用原 cohort 的前16题，两个 Base 模型各做两种提示条件，仍然
seed21、16K 上限，不更新权重，共64条生成：

- `no_think`：保留聊天模板、题目、逐步推理与 boxed 答案要求，只删除
  `conduct reasoning inside <think> and </think> and ` 这一小段文字。
- `plain`：裸题目加 `Solution:` 续写提示，同时移除聊天包装和环境指令。
  这是整个提示格式的替代条件，不能用它单独归因于某一个特殊 token。

只与第一轮相同16题的 Base 输出对比，不能拿16题消融直接对比32题基线。
诊断代码 v3 仅增加 no_think 分支和校验，7个单元测试通过；全库
517 passed、2 skipped。新代码未覆盖 v2。

## 完成记录

两轮共192条输出全部生成并用历史 grader 评分，无评分异常。
结果与限制见 [诊断结果](../results/2026-09-13-qwen06-truncation-diagnosis.md)。
完整原始证据已备份到本地 `/home/tyf/paper/outputs/qwen06-truncation-diagnosis`。
没有恢复评测队列或启动新训练。
