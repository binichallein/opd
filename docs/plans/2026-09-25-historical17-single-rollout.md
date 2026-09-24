# 旧版 B/C：停止与单回答配置准备

用户要求先确认 Step50 可恢复，再停止本轮训练，准备每题一条、batch32、200步；
新配置准备完报告，暂不开始训练。该要求覆盖此前 B/C 自动训练与评测授权。

## 原实验保全

旧目录：`runs/20260924v1_historical17_components_seed21_ml2`。
四rank Step50 Adam状态、scheduler/RNG和数据消费位置的CPU检查通过后，
2026-09-25 01:10北京时间发送SIGTERM，停止控制器及其训练进程组。
最后完成Step56，最近完整保存节点为Step50。C未开始，没有启动自动评测。
旧队列保留原始error143记录，不把人为停止伪装成训练正常完成或数值崩溃。

独立恢复验收目录：`runs/20260925v1_historical17_step50_resume_audit_ml2`。
仅从旧Step50做一次旧4题x8回答配置的Step51更新，并保存新完整状态；
调度器总步数仍为200，不改成51。原checkpoint加载后不删除，源文件前后SHA核对。
同时验收四rank优化器步数、模型更新、RNG/调度器、数据下一批及32条原始轨迹。
这是恢复测试，不是新n1配置训练，也不代表参数更新的bitwise复现。

## 新配置

新目录：`runs/20260925v1_historical17_components_n1_batch32_seed21_ml2`。
准备器：`scripts/prepare_historical_single_rollout.py`，只生成命令和run card，无启动分支。
冻结训练部署：`7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5`。

| 配置 | 旧版 | 新版 |
|---|---|---|
| 每步题目数 | 4 | 32 |
| 每题回答数 | 8 | 1 |
| 每步轨迹数 / PPO mini-batch | 32 / 32 | 32 / 32 |
| 总更新步数 | 200 | 200 |
| 学生 / 教师 | Qwen3-1.7B-Base / 公开Qwen3-4B-Base-GRPO | 不变 |
| 数据 | DAPO-Math-17k，同一train.parquet及SHA | 不变 |
| seed / 请求seed规则 | 21 / legacy逐请求21 | 不变 |
| prompt和EOS/mask | 旧版1.7B ChatML/think训练设置 | 不变 |
| 学习率 / PPO epochs | 2e-6 / 1 | 不变 |
| max prompt / response | 2048 / 16384 | 不变 |
| temperature / top-p | 1.0 / 0.9 | 不变 |
| GPU / actor micro-batch / vLLM显存比例 | 4 / 1 / 0.6 | 不变 |
| 完整checkpoint | 50/100/150/200，不删除 | 不变 |
| 可观测性 | 所有原始rollout、每5步诊断及位置统计 | 不变 |
| 后续评测协议 | 原历史grader，四benchmark分别完整n8评分 | 不变，未启动 |

B=`adv3/adv_only`：共享均值advantage，逐token ratio与归一化。
C=`joint3/joint_tokenmean`：联合block ratio，每block一个surrogate项除以有效token数。
两组以后均从原始学生独立初始化，不从刚停止的B Step50接续新采样配置。

## 对照边界

run card允许的差异只有每步题数、每题回答数和新输出目录；逐字段拒绝其他变化。
数据与运行代码manifest也与旧实验逐项比较，B/C保持同一训练配置及输入来源。
新配置总计6400次题目出现，旧版800次；这不代表6400道不重复题，也不代表遍历17k。
两者都是6400条训练轨迹，所以不能承诺训练耗时降为1/8。
本轮没有同时改为“不给请求固定seed”，没有修改旧prompt或修正历史EOS行为。
新B/C彼此可作同采样配置的消融，但与旧A/D比较混入了题目覆盖量的改变。

准备状态以远端`preparation.json`为准：必须`prepared=true`、
`training_started=false`、`launch_authorized=false`。不运行旧自动控制器。
