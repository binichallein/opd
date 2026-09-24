# 旧版1.7B / 4B-GRPO B、C消融启动

## 2026-09-24 22:29北京时间快照

自动队列已启动，仅ml2。B第一步GPU更新和保存验收通过，22:29:25启动B恢复探针。
正式200步尚未启动，C尚未启动；下述首步数字来自探针，不是正式训练结果或benchmark分数。

- 不可变运行代码：`7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5`。
- 控制器PID `3519314`，PPID1、独立SID3519314，服务端nohup；不依赖本地窗口。
- B恢复探针PID `3536814`。所有状态以远端实时文件为准，不按此快照重复启动。
- 远端根目录：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260924v1_historical17_components_seed21_ml2`。
- 控制日志 `controller.log`，实时状态 `queue_state.json`。

## 方法和配置

B=`adv3`：block3平均advantage，逐token PPO ratio，token归一化。
C=`joint3`：同样平均advantage，联合block PPO ratio，非空block surrogate求和后除有效token数。
C消除历史Block3的实际块长倍率，不是只把lr除3；尾部短块也校正。
不同reduction的loss绝对值不直接代表效果好坏，Adam更新也不等同于原始梯度缩放。

两组各200步，seed21，同一DAPO文件与历史原始学生初始化，分别保留50/100/150/200完整状态。
训练ChatML+think要求、未指定thinking模板kwargs；评测ChatML+关闭thinking。
逐请求重复seed21、原生EOS151643、历史EOS mask全部明确保留。
这是用户要求的历史协议消融，不推荐把这些已知局限当作新实验默认。
历史A/D不重训，150没有对应历史A/D；不将本轮称为四组同环境重新训练。

## 已完成验收

1. ml2真实环境97项相关CPU测试通过；新增数学测试覆盖梯度、裁剪、partial block、mask和stop-gradient。
2. 另24个CPU样例中，新代码legacy分支与冻结`0f9161f`的loss、指标、梯度逐元素完全一致。
   只证明loss函数回归，不证明完整训练bitwise重放。
3. 历史模型/数据artifact manifest通过；训练采样、micro-batch、学习率等与旧run card逐项匹配。
4. 真实collector核验643题训练输入；独立核验643题历史评测输入，明确记录训推差异。
5. B CPU Ray实际worker检查通过。
6. B探针Step1退出码0，checkpoint审计、四rank非零且有限Adam状态检查、32条轨迹审计通过。

## B探针首步

| 指标 | 值 |
|---|---:|
| 裁剪前grad norm | 121.130699 |
| PG loss | 0.687836 |
| PG clip fraction | 0.000496 |
| 平均响应长度 | 4630.25 |
| 长度截断 | 8/32 |
| 学生entropy | 0.067677 |
| 教师entropy | 0.053258 |
| Sign flip rate | 0.253073 |
| 非有限数值计数 | 全部0 |
| 32条中的不同输出数 | 4 |

首批prompt token multiset哈希为`00e902b1ea756ff9ec0cb6e86e8c1e0c17418ff6b448186026d0b9803ec5e402`，
与历史Block3 Step1一致；响应长度和师生熵也相同。历史没有保留原始训练rollout，不能声称完成逐token输出比对。
只有4种不同输出体现重复请求seed带来的退化采样，本轮有意保留并完整记录。
首批periodic-tail检测0、生成的新think标签0，不代表全训练无重复或不推理。

## 自动后续

B恢复探针通过 -> 从原始学生重新初始化正式B200 -> 全状态和6400条轨迹审计、热力图
-> B完整评测200/150/100/50 -> C独立保存/恢复探针 -> C200及相同评测。
每次评测643题x8=5144条，固定历史grader，四个benchmark分别报告Avg@8/Pass@8及格式/截断。
任何验收失败立即停队列，保留失败证据；不盲目重试或调参。

## 归档

代码和方案已推送GitHub `feat/qwen4-blockfirst`。启动元数据备份：
`/home/tyf/paper/outputs/historical17_components_20260924/startup_7bf5420`。
首批5份启动JSON的本地/远端SHA256已逐项核对；另外保存B探针诊断与验收文件。
完整模型状态和全部原始训练/评测轨迹保留ml2，不上传GitHub。本次不是权重的本地备份。
