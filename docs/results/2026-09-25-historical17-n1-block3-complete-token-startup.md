# 历史1.7B / 4B-GRPO：Block3完整评测与Token启动监督

## 范围与证据

这是已授权的32题位置x1条回答、100步、seed21实验，不是旧4x8实验或B/C消融。
学生为官方Qwen3-1.7B-Base，教师为公开lllyx/Qwen3-4B-Base-GRPO。
完整历史Block3 Mean包含共享advantage、联合block PPO ratio及历史block归一化。
旧训练/评测prompt及其差异、请求seed、mask和冻结运行代码保持不变。

运行目录：`runs/20260925v4_historical17_pair_n1_step100_save25_seed21_ml2`。
训练/评测部署：`7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5`；控制部署：`0baacaf3ec6c143f227e038f6d092a5b0f99d124`。
后台控制器PID3640782；本轮监督未修改训练代码、配置或重启任务。

截至北京时间2026-09-25 15:35，Block3正式100步及25/50/75/100四个checkpoint评测均完成。
四个评测在`evaluation_acceptance.json`中各自`passed=true`；总队列仍`complete=false`，
因为Token训练和四次评测尚未完成，不能把Block3完成写成整个对照实验完成。
每个checkpoint生成MATH500 4000条、AIME24 240条、AIME25 240条、AMC23 664条，
使用历史grader；结果按benchmark分别报告，不合并总分。

原始机器可读结果：[block3_complete.json](../../results/historical17_pair_n1_save25_20260925/block3_complete.json)。
本地验收与结果快照：`/home/tyf/paper/outputs/historical17_pair_n1_save25_20260925/v4/supervision_1540/`。
原始rollout、权重、完整日志和位置数组不进入Git。

## Block3结果

所有分数均为百分比。

| Benchmark | Step25 Avg@8 / Pass@8 | Step50 Avg@8 / Pass@8 | Step75 Avg@8 / Pass@8 | Step100 Avg@8 / Pass@8 |
|---|---:|---:|---:|---:|
| MATH500 | 47.00 / 82.40 | 39.68 / 85.00 | 43.18 / 84.40 | 56.40 / 87.20 |
| AIME24 | 5.00 / 26.67 | 5.83 / 30.00 | 3.75 / 20.00 | 8.33 / 23.33 |
| AIME25 | 2.50 / 10.00 | 0.83 / 3.33 | 2.08 / 10.00 | 5.83 / 23.33 |
| AMC23 | 18.37 / 49.40 | 20.63 / 55.42 | 20.63 / 57.83 | 26.51 / 60.24 |

Step100在四个benchmark的Avg@8均为本组最高，但AIME24的Pass@8最高在Step50，
不能声称Step100在所有指标均最好。曲线非单调，且长尾退化对部分评测轮次影响明显。
MATH500引擎长度截断率在Step25/50/75/100分别为11.675%/25.825%/17.625%/6.275%。
详细诊断见[长尾审计](2026-09-25-historical17-n1-eval-long-tail.md)。
在匹配的Token结果产生前，不下Block3优于Token的结论，不拿旧4x8结果替代新对照。

## Token启动

15:35自动进入Token流程，Ray CPU预检通过；15:36:03启动保存Step1预检，PID3864363。
15:43正在加载四rank模型，尚未完成首个更新，也未完成保存/恢复验收或启动正式训练。
预检先保存Step1、再从完整状态恢复至Step2，保持100步scheduler horizon。
正式训练仍从原始学生初始化；计划100步、保存25/50/75/100全部完整状态，随后评测100/75/50/25。

已逐键核对Block3与Token正式run card：差异仅variant/输出路径/实验名及block size、advantage mode。
两组都保留每步实际rollout、Step1与每5步完整诊断、位置热图和全部checkpoint。
上述为定时快照，实际任务状态必须重新读取服务器记录。

### 15:52预检更新

保存Step1预检退出0，完整四rank模型/优化器/scheduler/RNG/数据状态、32条轨迹与位置诊断通过验收。
退出后的DataLoader析构告警不等于更新失败；退出码和实物验收均已检查。
15:51:36自动启动恢复Step2预检，PID3878075，正式100步仍未开始。

预检首批32条与已完成Block3正式首批的题目、prompt token、response token、采样设置和mask逐条一致。
长度截断16/32、均长8525.40625、新生成think标签0；首次更新前的高截断不是本轮Token更新造成。
裁剪前梯度范数27.0606，PG loss0.505217，sign flip/weighted sign flip/leakage均为0。
student/teacher entropy为0.0470112/0.0402648，top16 overlap为0.635083；已记录非有限数计数均为0。
这是独立预检指标，不是正式训练的Step1，也不是benchmark结果；有限梯度不代表生成健康。

首批原始压缩轨迹、标量及位置数组备份于本地`supervision_1540/token_probe1/`。
Block3分榜结果的服务器、本地和Git摘要三份SHA256一致：
`b86acc67859f462c9e706a9cb44fe3305222ea4244480ec947ac0d7fe06a8c90`。
本地24项配置/队列/历史prompt/损失测试通过；默认Python的Torch链接失败，改用已有unsloth环境，
复用已有pytest运行，未安装包或更改ml2环境。所有更改仅涉及记录和结果摘要。
