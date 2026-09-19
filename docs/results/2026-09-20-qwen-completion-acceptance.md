# Qwen completion 三项验收

## 范围与当前状态

验证旧 Qwen4 预更新循环问题的协议修正，不是 Block3 涨点实验。
原始模型提示诊断、历史 grader 评分、两组真实更新与恢复验收均已完成。
工程验收通过，但教师适配仅有小样本支持，循环也未彻底消失，不能笼统称为
“三个问题全部解决”。旧 v1 队列保持停止，未启动新的正式 200 步训练。

- 学生：官方 ModelScope `Qwen/Qwen3-4B-Base`，revision
  `bbd6fc8d23e8788d987b7b970cbb7bd31c826e38`。
- 教师：沿用公共 `Qwen3-4B-Base-GRPO`，逐文件核验历史 SHA256 清单。
- 题目：旧 Qwen4 正式 Step1--4 的 16 个训练题，不是测试 benchmark。
- 每题两次，seed21/22；同题、同 seed、同采样参数比较三种提示与两个模型。
- 温度1、top-p0.9、top-k-1、上限16384，BF16 vLLM0.11 TP1，模型 EOS151643。
- 历史 grader SHA256：`04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
- 192 输出完整保存，输入/输出 SHA、token ID、logprob、终止原因全部核验；评分异常0。

## 原始模型对照结果

每行分母32。格式错误仅指历史定义的“没有 `\boxed`”，不代表数学正确性。
周期重复使用已有尾部周期检测器；这些是两个不同指标。

| 模型 | 提示 | 答对 | 截断 | 周期重复 | 缺少 boxed |
|---|---|---:|---:|---:|---:|
| 学生 | 历史 ChatML + think 要求 | 1 | 18 | 17 | 21 |
| 学生 | 裸题目 + Solution | 2 | 0 | 0 | 9 |
| 学生 | 最终 boxed completion | 4 | 1 | 1 | 0 |
| 教师 | 历史 ChatML + think 要求 | 6 | 2 | 0 | 2 |
| 教师 | 裸题目 + Solution | 5 | 0 | 0 | 0 |
| 教师 | 最终 boxed completion | 8 | 3 | 2 | 2 |

最终提示下，学生4个题至少一次正确，教师5个题至少一次正确。因此不能把
“8比4”解释为教师在两倍数量的题上会解题，更不能把32次输出当成32个独立题目。

### 结论边界

1. **教师适配：有限支持，非全面可靠性证明。** 最终提示没有观察到教师正确率
   优势丢失，但仅16题，教师24/32仍然答错且3/32截断。独立生成结果不能证明
   teacher-on-student-prefix 指导始终可靠。
2. **最终提示：学生输出明显改善，但不是消除所有循环。** 相比旧提示，学生
   截断18→1、周期重复17→1、格式错误21→0；不能只拿裸提示0截断代替最终提示结果。
   教师在最终提示下的循环多于裸提示，不能宣称修复对师生所有指标都单调有利。
3. **独立请求 seed：两组实际训练与恢复均通过。** 不再是仅修改推理诊断脚本；
   真实训练中每批32个独立seed，每题8条不同输出，跨方法和恢复后的种子计划均核验。

## 有界训练结果

2026-09-20 00:43:19北京时间，控制器记录 `complete`。每行32条实际训练输出；
四批共128条，原始seed21题目顺序、实际训推token、历史EOS mask和有限logprob
均通过检查。这里不做benchmark评分，也不以两步差异判定方法优劣。

| 方法 | Step | 截断 | 周期重复 | 缺少 boxed | 裁剪前 grad norm | 学生熵 | 教师熵 | Top-16 overlap | SignFlipRate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Token OPD | 1 | 0 | 0 | 1 | 4.42339 | 0.20898 | 0.17316 | 0.91432 | 0 |
| Token OPD | 2 | 4 | 4 | 3 | 1.28878 | 0.10832 | 0.08863 | 0.90974 | 0 |
| Block3 Mean | 1 | 0 | 0 | 1 | 5.42485 | 0.20898 | 0.17316 | 0.91432 | 0.16348 |
| Block3 Mean | 2 | 2 | 1 | 2 | 1.25157 | 0.14911 | 0.12028 | 0.88969 | 0.09410 |

没有非有限诊断值，但有低熵长尾重复，不能据此推断200步不会崩溃。
Block3第二步的两条截断题在原始模型诊断中也曾循环，Token也存在循环，
因此不能简单归因为Block3，也不能把2比4当作Block3有效的统计证据。

两组都已直接查看两步的四个rank：每rank的37个Adam状态从step1推进到step2，
LR2e-6；调度器同步推进，CPU/CUDA/NumPy/Python RNG均保存；dataloader已消费
题数从4推进到8。恢复日志确认加载model、optimizer、extra state，非仅加载权重。
原始轨迹、位置诊断和热力图保留。默认热图屏蔽少于8条轨迹的位置，灰色不是零熵。
另存 `figures_min1` 低计数诊断图，呈现稀少长尾；样本数仍应从NPZ的valid_count读取，
这些尾部不能视为稳定的总体趋势。只引用正确标注方法的PNG；沿用脚本的HTML介绍
有历史写死的“Block3”措辞，Token目录中的HTML介绍不作为方法身份依据。

首次更新前，两组32条输入、数据身份、seed、采样参数、完整输出token和mask均相同。
logprob数组28/32逐位相同，另4条共有6个位置不同，最大绝对差0.0142758；
这证明该批实际输出一致，不证明GPU浮点逐位确定性，也不是无中断恢复等价性证明。

四个训练进程均返回0；收尾日志有DataLoader worker被结束的警告，
已保留，不能描述成“日志没有任何异常”。它们没有阻止checkpoint与归档验收，
两组均成功恢复。无权限读内核日志，未单独证明该worker被结束的内核级原因。

## 训推一致性

`qwen3_completion_boxed_v1` 共用 `completion_math_prompt`；无 ChatML/think/BOS。
真实 MathEnvironmentManager 与 TrajectoryCollector 已逐题处理 MATH500500题、
AIME2430题、AIME2530题、AMC2383题，643个输入均与评测端 token ID 完全一致。
这是 CPU 输入协议验收，不是对这些 benchmark 做了新评分。

新训练 seed 规则为 `sha256_step_question_sample_v1`，以全局21、step、数据行编号、
题目哈希、样本序号和 turn 派生。实际 vLLM SamplingParams 按请求独立设置，
归档保留身份和实际 seed；评测仍使用历史 seeds21..28。
历史 Block3 joint-block PPO ratio、损失归一化、Qwen EOS mask 均未改动。

新评测必须显式传入 `--prompt-protocol qwen3_completion_boxed_v1`。
旧评测默认和旧正式队列刻意保持历史协议；不能直接重启旧 controller 来运行新对照。
本次有界 gate 不会自动启动完整 benchmark 或正式训练。

## 追溯

- 推理代码：`048f0f8acfaa74e9679e41c1d1c4acfdedf8a471`，独立 analysis deployment。
- 新训练代码：`0f9161f02f08287fb07f0375ad0a6bda81133ff0`，独立冻结 deployment。
- 原始推理：ml2 `diagnostics/20260920_qwen_completion_acceptance`。
- 完整评分：该目录 `audit_and_grades.json`；仓库只保存小型汇总 JSON。
- 有界训练：ml2 `runs/20260920v1_qwen4_completion_gate_seed21_ml2`。
- 控制器1128081于2026-09-19 23:45北京时间启动；run ID 用20260920命名。
- 只运行 Block3/Token 各 Step1保存、退出、恢复至Step2。初始权重相同且独立，
  数据、LR2e-6、4题×8轨迹、16K上限、microbatch1、vLLM0.6均相同。
- 所有探针权重、四 rank 训练状态、逐步 rollout 和位置诊断保留，禁止当作正式结果。
- 小型验收证据：[训练汇总 JSON](2026-09-20-qwen-completion-gate-summary.json)；
  [原始推理汇总 JSON](2026-09-20-qwen-completion-inference-summary.json)。
- 收尾复跑相关单元测试101项通过；原始资产和训练数据哈希在gate结束前再次核验。
- 收尾复查GPU compute-apps为空；此次验收已结束，没有后台正式训练。
