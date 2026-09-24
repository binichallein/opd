# 旧版1.7B / 4B-GRPO B、C消融监督

## 恢复验收与正式启动

2026-09-24 22:43:28北京时间，B (`adv3`) 保存/恢复探针验收通过，
控制器自动启动正式200步任务，PID `3548435`。控制器仍为 `3519314`。
运行代码仍是不可变部署 `7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5`。
本轮监督没有重启任务、修改训练参数、部署代码或更换模型。

截至22:55:49，正式任务已完成Step1/200，C仍排队。
正式命令为 `resume_mode=disable`、空resume路径，使用历史原始学生独立初始化，
不是接着探针Step2训练。50/100/150/200完整状态保存节点已核对。

恢复验收核对了两步checkpoint的四rank模型、Adam状态、RNG、scheduler及数据消费位置，
并确认恢复日志包含每个rank的模型、优化器、额外状态加载路径。
两步全部64条轨迹的输入顺序、历史prompt、采样seed、EOS mask和log-prob字段审计通过。
这是恢复工程验收，不是完整训练的bitwise复现证明。

## 正式首步验收

32条正式首批样本与探针Step1的题目元数据、prompt token、response token、
采样参数、完整训练mask、终止原因全部逐条一致，归档SHA与sidecar也一致。
正式首步裁剪前grad norm为121.130737，PG loss为0.687836，平均长度4630.25，
截断8/32，学生entropy为0.067677，sign flip为0.253073。
全部已记录数值有限，非有限计数均为0。原始轨迹SHA为
`ab3825da3ef75ee0ad149e3973592d8d6cd44e87767fd45d531145b14422be86`。
正式与探针的原始梯度末位及更新后ratio统计并非完全相同，故只报告输入/生成一致，
不声称参数更新bitwise一致，更不以首步验收代表200步稳定或benchmark提升。
详细记录见 `results/historical17_components_20260924/resume_supervision/formal_step1_observation.json`。

## 探针数值与输出风险

| 指标 | Step1 | 恢复后Step2 |
|---|---:|---:|
| 裁剪前grad norm | 121.130699 | 2.109889 |
| PG loss | 0.687836 | 0.059229 |
| 学生entropy | 0.067677 | 0.087738 |
| 教师entropy | 0.053258 | 0.074091 |
| Top-16 overlap ratio | 0.612845 | 0.649764 |
| Sign flip rate | 0.253073 | 0.122487 |
| 平均响应长度 | 4630.25 | 9242.75 |
| 截断 | 8/32 | 16/32 |
| 检测到周期性尾部 | 0/32 | 8/32 |
| 不同输出数 | 4 | 4 |

所有已记录数值有限，非有限计数、block ratio overflow/underflow计数均为0。
梯度有限不代表生成健康：Step2一个问题的末尾明确循环重复中文连接短语，另一个问题
达到16K上限但未命中短周期尾部检测。逐请求固定seed21使每题8份输出相同，
故Step2实际为4道题中2道截断、其中1道命中周期性尾部，不能当成32个独立样本。
也不能仅凭两步、不同题目的观察认定是B方法导致collapse。

只读检查历史A/D的原始 `logs/nohup.log` 得到如下对照（历史日志精度保留3位）：

| Step2 | 历史Token A | 历史Block3 D | 当前B恢复探针 |
|---|---:|---:|---:|
| 截断比例 | 0 | 0.5 | 0.5 |
| 平均响应长度 | 2812.25 | 8701.0 | 9242.75 |
| 裁剪前grad norm | 6.025 | 8.682 | 2.109889 |

历史路径分别为 `20260712v1_token_opd_replication_seed21_ml2/token_opd` 与
`20260711v2_block3_replication_seed21_ml2/block3_mean`，均位于远端 `runs/`。
三者Step1截断均为0.25、平均长度均为4630.25。历史未保留原始训练输出，
不能从相同截断率推断循环文本也相同，或将跨运行环境的前两步视为最终因果结论。

本轮明确复刻旧协议，保留ChatML/think要求、旧EOS和重复请求seed；不在消融中途
通过改prompt、stop token或采样seed来改善这些指标。后续需连同完整评测报告局限。

探针退出阶段日志有 `Exception ignored in MathMultiProcessEnv.__del__` 和
`DataLoader worker ... is killed by signal: Killed`。探针及checkpoint审计退出码均为0，
完整状态验收通过，自动队列继续前进；未观察到CUDA OOM或非有限更新。
保留该退出清理警告，不把它隐去，也不据此盲目重启。

## 留存与后续

恢复验收、四rank优化器审计、64条轨迹审计、checkpoint验收、两步诊断分别保存在
`results/historical17_components_20260924/resume_supervision/`。
相同五份文件本地备份目录：
`/home/tyf/paper/outputs/historical17_components_20260924/supervision_20260924_2244`。
本地/远端SHA256逐项相同。完整checkpoint和原始轨迹保留ml2，本次未复制权重到本地。
另保存探针Step2位置NPZ到上述本地目录并核对远端SHA；其16K位置统计数组均为有限值。
热力图由原队列在正式训练结束后生成。本轮不把探针热图冒充正式训练热图。

自动顺序不变：B200 -> checkpoint/轨迹审计和热力图 -> B完整评测200/150/100/50
-> C独立探针、C200及相同评测。当前没有新增benchmark分数。
