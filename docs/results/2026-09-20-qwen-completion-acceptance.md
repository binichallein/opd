# Qwen completion 三项验收

## 范围与当前状态

验证旧 Qwen4 预更新循环问题的协议修正，不是 Block3 涨点实验。
原始模型提示诊断、历史 grader 评分已完成。真实训练更新与恢复验收正在进行，
尚不能写成三项全部通过；旧 v1 队列保持停止，未启动新的正式 200 步训练。

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
3. **独立请求 seed：第一步真实 GPU 更新已通过；完整恢复结果待补。**

## 有界训练进度

Block3 Step1 已完成真实更新并保存完整 checkpoint。32条轨迹使用32个不同seed，
每道题的8次输出均不同；截断0、周期重复0、格式错误1。原始seed21题目顺序、
实际训推token、历史EOS mask和有限logprob均通过检查。
裁剪前grad norm5.42485，学生/教师熵0.20898/0.17316，Top-16 overlap0.91432，
sign-flip rate0.16348。没有记录非有限数值。不能据此推断200步不会崩溃。

第一进程已退出；2026-09-20 00:01:07北京时间启动恢复进程1140852。
已直接查看rank0保存的Adam状态：37个状态全部step1，LR2e-6；调度器step1、
四类RNG和dataloader已消费4题均存在。其余rank及Step2需在恢复完成后统一复核。

## 训推一致性

`qwen3_completion_boxed_v1` 共用 `completion_math_prompt`；无 ChatML/think/BOS。
真实 MathEnvironmentManager 与 TrajectoryCollector 已逐题处理 MATH500500题、
AIME2430题、AIME2530题、AMC2383题，643个输入均与评测端 token ID 完全一致。
这是 CPU 输入协议验收，不是对这些 benchmark 做了新评分。

新训练 seed 规则为 `sha256_step_question_sample_v1`，以全局21、step、数据行编号、
题目哈希、样本序号和 turn 派生。实际 vLLM SamplingParams 按请求独立设置，
归档保留身份和实际 seed；评测仍使用历史 seeds21..28。
历史 Block3 joint-block PPO ratio、损失归一化、Qwen EOS mask 均未改动。

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
