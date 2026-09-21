# Qwen8 to Qwen1.7 Capability Diagnostics Implementation Plan

**Goal:** 按用户顺序，只验收8B-Base对1.7B-Base、8B指令版对1.7B指令版，不启动训练。

**Architecture:** 新独立只评测队列；复用上一轮冻结选题、历史grader、采样与统计函数，
但不更改旧程序、旧产物或训练runtime。所有新资产来自官方ModelScope并固定revision。

**Tech Stack:** Python、vLLM0.11、Transformers4.57.6、pytest、现有不可变部署和后台控制器。

## 已确定的实验设计

- 先Base：`Qwen/Qwen3-8B-Base` -> `Qwen/Qwen3-1.7B-Base`。
- 后指令版：官方仓库名 `Qwen/Qwen3-8B` -> `Qwen/Qwen3-1.7B`，两者支持thinking开关，
  本次显式 `enable_thinking=False`，不用不存在的`-Instruct`别名。
- ml2唯一执行机器；全新根目录 `runs/20260921v1_qwen8_to17_diagnostics_ml2`。
- 复用上一轮已封存的64题选题及排除清单，不根据已有分数重新抽题。
  每个模型64题x2独立答题；每组自己的学生sample0前缀取固定前32题，师生各续写2次。
  两组各384条，共768条全新生成；旧8B独立结果保留为复核参考，不伪装成本轮新结果。
- Base沿用完全相同的completion boxed提示；指令版使用相同题意和boxed指令、原生聊天模板，
  `add_generation_prompt=True, enable_thinking=False`。两种prompt分开解释，不称纯模型规模消融。
- 温度1/top_p0.9/top_k=-1、16K总response预算、同题同sample的历史确定性seed保持。
  指令版使用真实原生EOS/停止ID，Base仍为151643；不修改任何模型tokenizer文件。
- 指令版先做GPU小样本非thinking检查，保存原始输入、输出、token与结束原因。
  模板可能在输入中放空think块；必须将输入模板标记和新生成think标签区分记录。
  检查失败则停该组，不以删除输出think内容掩盖失败，不偷偷调整提示或采样。
- 两个模型必须在各组内使用完全相同的token输入、前缀、seed和预算。
- 各组单独按历史grader计算平均正确率、配对题目bootstrap95%区间、截断、周期重复、
  无有效boxed率和新生成think标签。不是四benchmark完整eval。
- 沿用预冻结能力标准：独立答题至少+5pp且配对CI下界>0，续写差值非负，
  以及旧截断/重复/格式门槛。小样本证据不足记inconclusive，不自动追加直到通过。
- Base能力结果不影响继续测试指令组；工程/资产完整性异常则停止并保留证据。
- 无论两组结果如何，均不启动OPD训练、不修改旧验收阈值、不覆盖旧实验。

## 实现与验证任务

1. 新 `scripts/prepare_qwen17_pair_assets.py` 与资产测试：固定三个新模型的完整revision、
   大小和SHA，排他下载/断点续传；复用已验收8B-Base但不改写。支持1.7B单文件权重。
2. 新 `scripts/qualify_qwen17_pairs.py` 与测试：封存旧选题引用、新pair manifest、
   Base/chat prompt、各组学生前缀、真实EOS、非thinking GPU检查、全量原始轨迹与复算验收。
3. 新 `scripts/run_qwen17_pair_diagnostics.py` 与测试：仅串行验收，无训练调用；
   nohup服务端进程、不可变control release、GPU空闲检查、独立tmpfs缓存和失败保留。
4. 先执行CPU测试和独立代码复核，再实际ml2环境验证，固定提交并部署。
5. 启动一次队列，检查PID脱离SSH、首任务日志及资产进度；记录命令、路径和实际阶段。

## 状态

- [x] 用户确定两组及测试顺序；旧8B对4B验收结束、训练未启动、GPU空闲。
- [x] 官方ModelScope模型名称与非thinking用法已核验。
- [x] 新资产与验收代码实现、测试、复核；ml2实际环境264项CPU测试通过。
- [x] 不可变部署与队列启动核验；19:39北京时间启动PID1872503，PPID1/SID1872503。
- [ ] Base组结果。
- [ ] 指令组GPU非thinking验收和能力结果。

启动证据见[启动记录](../results/2026-09-21-qwen8-to17-diagnostics-startup.md)。
状态勾选只代表工程预检和后台启动完成，不代表模型能力已经验收。
