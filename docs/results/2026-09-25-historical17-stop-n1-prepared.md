# Step50恢复验收、停止与新配置准备

2026-09-25 01:30北京时间最终检查通过。旧训练及自动队列已停止，
独立恢复测试也已退出，四张GPU均4MiB、利用率0%。新B/C均未启动。

## Step50确实可恢复

先在CPU读取四rank的Adam动量、scheduler、RNG和dataloader状态，确认步数50；
随后01:10按用户要求SIGTERM停止旧控制器3519314及训练进程组3548435。
停止前最后完成Step56，旧目录最后完整保存节点为Step50，C从未正式启动。
error143保留为人为终止证据，不描述为训练崩溃。

在独立目录运行旧4题x8回答配置的Step50->51测试，scheduler总步数仍为200。
日志明确显示global step恢复为50，四rank分别读取model、optim和extra_state。
随后完成Step51更新，保存完整checkpoint并主动退出，训练退出码0。
四rank的新Adam状态有限且步数51，scheduler与数据消费位置也为51，模型shard均发生更新。
恢复后32条轨迹的题目顺序与原正式Step51一致，旧prompt/seed/EOS mask约束通过。
原Step50全部22个文件恢复前后SHA256一致，未删除、覆盖或替换。

本次只证明训练状态恢复和一次更新可执行，不证明bitwise复现，也不是新的benchmark结果。
恢复步骤日志的grad norm为8.629、平均长度603、截断0；这些是三位小数日志值，
不能代表新n1配置的稳定性。新n1没有进行GPU训练或推理验收。

## 新配置仅准备

200步；32个prompt位置/步，每位置生成1条，PPO mini-batch仍32。
只改变`train_batch_size:4->32`、`rollout_group_size:8->1`及输出目录。
其余run card字段、数据/代码manifest与旧B/C逐项一致，包括seed21及旧逐请求seed规则。
不改变prompt，不改loss；保留50/100/150/200完整状态、全部原始rollout及原过程指标。
两组计划从原始1.7B-Base独立初始化，不在旧Step50上中途改变采样配置。
[完整配置与对照边界](../plans/2026-09-25-historical17-single-rollout.md)。

准备器/恢复验收代码commit为`6ab2b25f4724ea55627fd436d006c47bc2740bd2`；
实际训练代码仍冻结于`7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5`，未修改旧deployment。
本地37项相关测试通过，两份远端命令通过bash语法检查和逐字段配置验收。
测试使用已有unsloth Python加载Torch，并复用本机pytest；未修改环境依赖。

## 证据与备份

- 原实验：ml2 `runs/20260924v1_historical17_components_seed21_ml2`。
- 恢复测试：ml2 `runs/20260925v1_historical17_step50_resume_audit_ml2`。
- 新配置：ml2 `runs/20260925v1_historical17_components_n1_batch32_seed21_ml2`。
- 上述`runs/`均相对于`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
- 本地验收文件及恢复测试raw轨迹：`/home/tyf/paper/outputs/historical17_stop_n1_20260925`。
- Git精简证据：`results/historical17_stop_n1_20260925/`，不包含权重或raw轨迹。

完整原Step50和恢复测试Step51权重、旧正式rollout均留在ml2；本轮没有将权重复制到本地。
`preparation.json`明确`training_started=false`、`launch_authorized=false`。
后续启动需用户新指令，不能重新运行旧自动控制器或把恢复探针计入正式实验。
