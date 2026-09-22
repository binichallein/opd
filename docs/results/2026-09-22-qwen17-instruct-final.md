# Qwen3-1.7B 指令版对 8B：最终配对结果

评测于2026-09-22 20:29:47北京时间全部完成：原始学生、Block3 Mean和Token OPD
各Step50/100/150/200，共9个模型视图。以下每个benchmark独立计分，不计算macro。

## 模型与对照

| 项目 | 本轮设置 |
|---|---|
| 学生 | 官方ModelScope `Qwen/Qwen3-1.7B`，revision `4855588ea1a12789f2e965e5f52a9e4a24c94b2a` |
| 教师 | 官方ModelScope `Qwen/Qwen3-8B`，revision `26028140be3ee69b82b1d1450179ab71bb1121b9` |
| 模型类型 | 两者都是后训练/指令版，官方名称不带`-Instruct`；不是Base版本 |
| 初始化 | 两组独立从同一原始学生初始化，不从probe或另一组权重续训 |
| 数据 | 同一DAPO派生文件，17,917行题答记录重复100次，共1,791,700行；精确唯一题面17,237个 |
| 数据SHA256 | `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf` |
| 训练 | seed21；每步4题，每题8条；200步；lr=2e-6；最大输出16,384 tokens |
| 显存配置 | 四张A100 80GB；actor microbatch=1；vLLM memory utilization=0.6 |
| 权重保留 | 每组Step50/100/150/200，完整模型、优化器与恢复状态；不自动删除 |
| 冻结训练/评测代码 | `be736b5fac4f26ff59f4e2c21abafc456c2503f6` |
| Token恢复控制器 | `cd2d1bde0a25844361f05a8230978eb3bd37dedb`，没有替换冻结训练代码 |

提示协议为`qwen3_native_chat_no_thinking_boxed_v1`：题面后追加
`Please solve the problem step by step and put the final answer in \boxed{}.`，
再使用原生chat template，显式`enable_thinking=False`。训练和评测使用同一函数，
核验token IDs；EOS=151645，停止集合=[151645,151643]，使用有效长度mask。

训练题目顺序和请求seed逐项配对核验；模型更新后生成内容不同是预期行为。
Block3是历史实现：共享mean advantage、联合block PPO ratio和block损失归一化。
本轮不能解释为“只改变advantage平滑”的纯消融。

原队列在Token正式训练的Ray启动阶段失败，尚无正式更新；该失败现场保留。
恢复队列复用已验收的Block3四次评测及原始学生评测，没有重跑或改写它们。

## 完整评测

MATH500=500题，AIME24=30题，AIME25=30题，AMC23=83题。每题8次，
seed21至28，temperature=1，top_p=0.9，输出上限16,384，历史grader不变。
每个模型5,144条输出，共46,296条。全部保留原始文本、token IDs、停止原因与评分。

以下单位均为百分比。Avg@8是8次回答的平均正确率；Pass@8是至少答对一次的题目比例。
`student_base`只是归档里的“训练前学生”标签，不代表它是Base模型。

### Avg@8

| 模型/Step | MATH500 | AIME24 | AIME25 | AMC23 |
|---|---:|---:|---:|---:|
| 原始学生 | 71.45 | 10.00 | 10.42 | 40.51 |
| Token 50 | 72.48 | 13.33 | 8.75 | 45.48 |
| Block3 50 | 71.50 | 12.50 | 9.58 | 43.67 |
| Token 100 | 72.93 | 15.00 | 12.08 | 43.98 |
| Block3 100 | 72.55 | 16.25 | 13.33 | 44.43 |
| Token 150 | 72.48 | 15.42 | 10.00 | 43.83 |
| Block3 150 | 72.63 | 13.33 | 12.08 | 46.84 |
| Token 200 | 72.13 | 14.17 | 12.08 | 43.07 |
| Block3 200 | 72.23 | 16.67 | 12.92 | 43.37 |

### Pass@8

| 模型/Step | MATH500 | AIME24 | AIME25 | AMC23 |
|---|---:|---:|---:|---:|
| 原始学生 | 91.20 | 26.67 | 23.33 | 69.88 |
| Token 50 | 90.60 | 33.33 | 26.67 | 73.49 |
| Block3 50 | 89.00 | 26.67 | 16.67 | 73.49 |
| Token 100 | 89.00 | 30.00 | 30.00 | 69.88 |
| Block3 100 | 91.00 | 36.67 | 33.33 | 68.67 |
| Token 150 | 90.60 | 30.00 | 26.67 | 71.08 |
| Block3 150 | 91.60 | 30.00 | 36.67 | 74.70 |
| Token 200 | 89.60 | 43.33 | 30.00 | 75.90 |
| Block3 200 | 89.80 | 36.67 | 30.00 | 74.70 |

### Step200格式与截断

| Benchmark | Token缺boxed | Block3缺boxed | Token引擎截断 | Block3引擎截断 |
|---|---:|---:|---:|---:|
| MATH500 | 0.125% | 0.150% | 0.175% | 0.150% |
| AIME24 | 1.667% | 2.083% | 1.667% | 2.500% |
| AIME25 | 0.417% | 0.000% | 0.417% | 0.000% |
| AMC23 | 0.602% | 0.000% | 0.602% | 0.000% |

历史format error仅检查回答中是否缺少字面`\boxed`标记，不是完整LaTeX语法验证，
也不代表数学答案错误。截断使用引擎`finish_reason=length`。所有9次评测的
generated think tag计数均为0；这不等于模型没有进行任何推理。

## 结论边界

- Step200 Block3减Token的Avg@8差值依次为+0.10、+2.50、+0.83、+0.30个百分点，四项均正。
- Step200 Pass@8分别为+0.20、-6.67、0、-1.20个百分点，并非全面提升。
- 其他checkpoint有反例：Step50 Block3三项Avg@8较低；Step150 AIME24较低。
  MATH500最高Avg@8出现在Token Step100，不能只报有利checkpoint。
- 两组Step200的四项Avg@8都高于原始学生，但MATH500 Pass@8均低于原始学生。
- 只有一个训练seed，四个checkpoint不是四次独立重复；当前结果不能证明统计显著性、
  普遍优越性或具体机制。已知训练池重复和历史数据重叠风险没有因本轮换模型而消失。

## 归档入口

- [机器可读汇总与全部配置](../../results/qwen17_instruct_20260922/final_20260922_2316/results.json)
- [36行分项指标CSV，保留未四舍五入数值](../../results/qwen17_instruct_20260922/final_20260922_2316/metrics.csv)
- [原始服务器评测验收清单](../../results/qwen17_instruct_20260922/final_20260922_2316/source_evaluation_acceptance.json)
- [本地归档校验清单](../../results/qwen17_instruct_20260922/final_20260922_2316/backup_acceptance.json)
- [Block3过程热图](../../results/qwen17_instruct_20260922/final_20260922_2316/figures/block3_mean/diagnostics.html)
- [Token过程热图](../../results/qwen17_instruct_20260922/final_20260922_2316/figures/token_opd/diagnostics.html)

本地原始结果目录：`/home/tyf/paper/outputs/qwen17_instruct_20260922/final_20260922_2316/`。
包含两组共12,800条训练轨迹、400步诊断快照、全部46,296条评测输出、评分与配置。
原始轨迹按服务器验收清单核对SHA256；其他本地文件另记哈希，便于后续完整性检查。
Git只保存汇总、图表和校验清单，不提交原始JSONL、日志、模型权重或训练parquet。
本轮结果备份不包含权重和训练parquet，它们仍保留在ml2，不能把本次归档称为完整模型备份。

源目录为`runs/20260921v1_qwen17_instruct_blockfirst_seed21_ml2`和
`runs/20260922v1_qwen17_instruct_token_recovery_seed21_ml2`；按验收清单区分两者。
中断的旧本地快照保留，不覆盖其评测验收状态，不作为最终结果入口。
