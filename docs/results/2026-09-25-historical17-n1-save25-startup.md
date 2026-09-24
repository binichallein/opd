# 历史1.7B / 4B-GRPO：32x1、100步、四权重启动记录

## 最新授权

本轮不是B/C消融。官方Qwen3-1.7B-Base学生与公开lllyx/Qwen3-4B-Base-GRPO教师，
先完整历史Block3 Mean，再Token OPD；各100步、seed21、32个prompt位置各生成1条。
旧训练/评测prompt及其差异、请求seed21、旧mask、DAPO文件、学习率2e-6和16K长度不变。
完整Block3仍包含共享advantage、联合block PPO ratio、历史block归一化，不是只平滑advantage。

用户补充：每25步保存，保留25/50/75/100全部完整恢复状态，不删除；初始学生已经评过，不再评测。
自动顺序为Block3保存/恢复验收、正式100步、完整评测100/75/50/25，随后Token同样流程。
只新增八次checkpoint评测；四榜MATH500、AIME24、AIME25、AMC23分别按历史grader计Avg@8和Pass@8。
原始学生不在新队列中，不能将已有基线说成本轮重新生成的结果。

## 启动与修复

- v3控制器3639963在北京时间02:21的CPU Ray socket长度检查被拦截：109字节超过107限制。
  未执行GPU训练/保存探针。原失败状态及日志保留，没有覆盖或自动重试。
- v4仅缩短Ray gate临时目录，并在预检时覆盖gate/train两种嵌套路径；不修改冻结训练部署。
  对应回归测试通过。配置记录也明确取消初始学生评测。
- v4控制器于北京时间02:25后台启动，PID3640782，PPID1、SID3640782，nohup/setsid确认。
- 控制代码：`0baacaf3ec6c143f227e038f6d092a5b0f99d124`。
  冻结训练/评测运行代码：`7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5`。
- Ray2.55.1实际CPU worker测试通过：导入1.72秒，启动与worker测试18.72秒；未修改库超时。
- Block3探针1于02:27:34启动，PID3641682。其Ray实例成功启动，正在构建原DAPO任务索引。
  此条记录不代表正式100步已开始，也不代表恢复门禁已通过；实际进度以LIVE记录为准。

### 02:43监督更新

探针1完成真实更新、保存和退出，exit0；完整四rank状态、Adam矩、scheduler/RNG/数据位置、
标量和位置统计、32条原始轨迹均通过CPU审计。实际输入顺序与32x1预先记录完全一致。
首批有32个不同物理数据行位置、31种不同输出；不是旧4x8重复采样。

| 探针Step1指标 | 数值 |
|---|---:|
| 轨迹数 / 每题回答数 | 32 / 1 |
| 长度截断 | 16/32 |
| 周期尾部诊断命中 | 10/32 |
| 新生成think标签 | 0/32 |
| 平均输出token数 | 8525.40625 |
| 裁剪前梯度范数 | 62.1910 |
| Sign flip rate | 9.3359% |
| Weighted sign flip rate | 5.7287% |

非有限advantage/entropy/block ratio计数和ratio上溢/下溢计数均为0。
生成发生在首次权重更新前，因此高截断及重复不能归因于本轮Block3更新。
这是探针，不是正式训练曲线，更不是benchmark评测。有限梯度也不能证明生成健康。

02:43:34自动启动探针2，PID3655323，指定从探针Step1恢复，仍保持100步scheduler horizon。
尚未确认恢复验收完成，正式100步尚未开始；Token及八次完整评测仍在队列内。

### 02:57后监督更新

恢复验收已通过：从Step1加载四rank模型/优化器/额外状态，实际推进并保存Step2后退出0。
四rank scheduler、RNG和数据位置均核验通过；Adam矩有限、非零，原Step1文件hash保持不变。
两步共64条原始轨迹的来源顺序、seed、prompt、token及mask核验通过。
探针Step2截断15/32、周期尾部9/32，梯度范数13.6124；仍是探针数据，不能标成正式Step2。

02:57:26自动开始正式Block3任务，PID3666134，父进程为后台控制器3640782。
正式run card核对：100步、32x1、seed21、完整legacy Block3 Mean，保存25/50/75/100；
`resume_mode=disable`、`resume_from_path`为空，从原始学生初始化，未继承探针权重。
Token训练/评测及Block3四次评测保持自动排队；原始学生评测明确排除。
本次监督没有更改任何训练参数、prompt或运行代码，没有重启任务。

### 03:17正式训练监督

正式训练完成Step3/100，四卡在运行，所读训练日志未见OOM、Traceback或ActorDiedError。
正式Step1的32条实际prompt和生成response token序列逐条匹配原始初始化探针1。
该匹配不代表优化更新逐位相同：正式Step1更新后的部分统计与探针存在浮点差异，
正式Step2也不应与探针Step2混用。

| 正式步骤 | 裁剪前梯度范数 | 截断率 | 平均输出token数 | 单步秒数 |
|---|---:|---:|---:|---:|
| 1 | 62.191 | 16/32 (50.0%) | 8525.406 | 247.133 |
| 2 | 23.988 | 17/32 (53.125%) | 9138.844 | 197.168 |
| 3 | 11.917 | 40.6% | 7606.531 | 201.993 |

上表梯度、长度、耗时及Step3截断率来自控制台舍入值；前两步截断数来自原始轨迹。
前两步累计截断33/64 (51.5625%)，周期尾部诊断分别10/32、13/32。
正式Step1完整诊断：student/teacher entropy为0.0470112/0.0402648，top16 overlap为0.635083，
student/teacher overlap mass为0.997190/0.997594，sign flip为9.3359%，weighted sign flip为5.7287%，
normalized leakage为1.01024；非有限advantage/entropy/block ratio及ratio上溢/下溢计数均为0。
完整分布/位置诊断按Step1及每5步保存，不把没有完整快照的Step2/3补写成已观测值。

前两步正式64条原始轨迹通过source顺序、32x1、seed、prompt/token/mask和SHA校验。
原始压缩轨迹、校验文件及Step1位置快照已备份到本地`v4/supervision_0300/`，
其中正式Step2原始文件位于子目录`formal_step2/`，避免覆盖同名Step1文件。
小型审计记录位于[监督证据](../../results/historical17_pair_n1_save25_20260925/supervision_0300/)，
原始轨迹和位置数组不进入Git。正式Step3目前仅核对控制台，未纳入前两步原始轨迹审计报告。

结论仅为恢复、训练及留存链路正在按配置推进，不能由有限梯度推出生成健康。
高截断/重复在首次更新前已经出现，不能据此断言由本轮Block3学习导致；仍需观察后续趋势。
尚未到首个正式保存节点25，Token及八次完整评测未开始；不重复评测初始学生。

## 验证与保留

53项新配置/队列/历史协议/损失相关测试及4项原block监督测试通过，Python编译检查通过。
实际冻结sampler已检查32x1输入计划：100步、3200个物理行位置。数据文件1791700行，
含原17917题的历史展开；不宣称3200道去重题，也不宣称与旧4x8有同样的prompt覆盖。
两组正式训练均必须独立从原始学生初始化，不能从探针或另一组权重继续。

服务器目录：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260925v4_historical17_pair_n1_step100_save25_seed21_ml2`。
本地证据：`/home/tyf/paper/outputs/historical17_pair_n1_save25_20260925/v4/`；失败记录另存`failed_v3/`。
已核对本地与远程preparation、acceptance、queue manifest的SHA256一致。
准备配置SHA256：`dce9210264b0786f50e78cc80c0c140e0312f35725faca0af3f07ced82a96fb0`。
每步原始rollout、各rank恢复状态、interval5诊断及位置热图按方案保留；权重和原始轨迹不入Git。

尚无本轮训练后的benchmark分数，不能据此判断Block3优于Token。
