# Qwen4 Completion Block3 评测进展

快照时间：2026-09-20 13:02 北京时间。仅 ml2；原训练、权重和原始结果不变。
本记录是阶段快照，不代表整个队列完成；操作前必须查询服务器实时状态。

## 执行状态

- 控制器于11:58:46启动，PID1502108，服务端 nohup，不依赖本地 Windows。
- 控制版本：`645a2f391f96311fe3ca3d85d0af44c0e877acbd`。
- 冻结训练/推理版本：`0f9161f02f08287fb07f0375ad0a6bda81133ff0`。
- Step200：12:01:14开始，12:39:43评测进程退出0；后续完整性验收通过。
- Step150：12:42:09开始，PID1511019；本次巡检时已完成MATH500生成，正在AIME24生成。
- 后续为Step100、Step50；当前队列不启动原始学生评测或Token训练。
- 不重复启动控制器，不覆盖已完成或正在写入的输出。

## Step200 独立分数

| Benchmark | 题数 | Rollout数 | Avg@8 | Pass@8 | 缺boxed比例 | 引擎截断率 |
|---|---:|---:|---:|---:|---:|---:|
| MATH500 | 500 | 4000 | 75.40% | 90.60% | 0.90% | 0.90% |
| AIME24 | 30 | 240 | 13.33% | 33.33% | 12.08% | 11.25% |
| AIME25 | 30 | 240 | 13.75% | 30.00% | 4.17% | 3.33% |
| AMC23 | 83 | 664 | 46.23% | 74.70% | 4.52% | 4.07% |

Avg@8是每题8次采样正确率的题间平均；Pass@8是8次中至少答对一次的题目比例。
各benchmark独立统计，不汇报macro或混合总分。
缺boxed比例来自历史字段format_error_rate，仅衡量缺少boxed，不等于全部格式错误；
实际截断率按引擎length终止原因统计，不能把两者等同。
本次新completion协议尚缺初始学生和Token对照，不能把绝对分数写成Block3方法增益。

## 协议与验收

- 学生为官方Qwen3-4B-Base，教师为原有公开Qwen3-4B-Base-GRPO；
  评测训练后的Block3 Mean Step200，不是原始学生或教师。
- `qwen3_completion_boxed_v1`：裸题目、step-by-step与boxed指令、Solution结尾，
  不使用ChatML，不要求think标签，enable_thinking=false。
- 全部643题的实际输入token、prompt、EOS151643、无额外stop token均通过验收。
- n=8，seed21..28，temperature=1.0，top_p=0.9，max_tokens=16384；
  历史四卡独立TP1评测，vLLM memory=0.9。
- 历史external grader SHA256：
  `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
- Step200共5144条原生rollout，32个archive文件；题目、seed覆盖、评分与原始轨迹一致性通过。
  保存输入/输出token、文本、seed与真实终止原因。数据、模型和grader hash按控制器契约核验。
- 合并及生成日志仍有tokenizer regex提示；本轮不热修改tokenizer。
  全部基准输入token一致性已核验，不据此声称该警告对任何输入都无影响。

## 证据位置

远程根目录：
`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2`

- `launch.json`：命令、PID、源码与部署SHA256。
- `queue_state.json`：实时阶段；`per_benchmark_results.json`：仅已验收权重的独立结果。
- `evaluations/block3_mean_step200/acceptance.json`：完整验收及产物hash。
- `evaluations/block3_mean_step200/outputs/rollout_archive/`：原生轨迹。
- 同名JSON记录本次精选元数据与未四舍五入的分数；原始轨迹、日志、权重不入Git。
