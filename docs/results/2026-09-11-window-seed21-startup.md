# Random3 / Sliding3 seed21 启动记录

本记录只描述实现与启动验收，不是方法有效性的实验结论。
执行范围以 [v2 协议](../plans/2026-09-11-sliding-window-opd-validation-v2-seed21.md)为准：
仅 ml2、Random3 与 Sliding3、各 seed21 / 200 steps，完整 Step50/100/200 四任务 n=8 评测。

## 固定实现

- Runtime commit：`fbad852a638de18e20d571a061b2be0437942a38`。
- 代码：`feat/ml2-block3-replication`，基于固定 Revisiting OPD 上游及版本化补丁。
- 本地测试：`460 passed, 2 skipped`；两项跳过源于测试环境没有 matplotlib。
  独立绘图环境已用合成诊断 fixture 完成五张 PNG 的渲染检查；不是实际实验图。
- ml2 训练环境没有安装或升级绘图依赖，绘图使用独立环境。
- Random3 的真实 Step1/2 快照也已生成五张诊断图及 HTML；人工检查位置热图，
  确认未覆盖位置为灰色缺失区。图位于恢复检查目录的 `gate_figures/`，不是正式结果。
- 原始模型、rollout、日志及 checkpoint 保留在 ml2，不上传 Git。

## 启动失败与隔离

`20260911v2_sliding_window_probe_seed21_ml2/random3` 在 Ray 初始化时退出 1：
缓存目录拼接后的 Unix socket 路径超过 Linux 的 107-byte 限制。
该尝试未执行 optimizer step，不是 OOM。保留整个失败目录及日志。

新尝试前缀为 `20260911v2r1`，仅缩短运行时缓存路径，不改变数学定义、
模型、数据、batch、精度、学习率、采样或最大长度。
缓存分别为 `/limx_embap/tos/wr/3r` 和 `/limx_embap/tos/wr/3s`。

以下路径以 `ROOT=/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd` 为前缀。

## Random3 恢复检查

目录：`$ROOT/runs/20260911v2r1_sliding_window_probe_seed21_ml2/random3`。

Step1 进程 PID `3052518` 正常退出 0，旧审计和新增 window 审计均通过。
恢复至 Step2 的进程 PID 为 `3065251`，于 `2026-09-11T10:08:28Z` 正常退出 0；
`probe-audit` 的旧审计与 window 审计均通过，issues 为空。

Step1 的已读取证据：

- `window_offset=2`，与独立 PCG64(seed=910021) 的第一抽一致；第二抽应为 1。
- 数据 checkpoint 的 `samples_yielded=4`、`snapshot_step=1`。
- 四 rank 均保存 model / optim / extra_state 文件；实际读取 optimizer，
  每份 29 个状态项均包含一阶/二阶动量，更新计数均为 1。
- 实际读取四份 extra_state：scheduler `last_epoch=1`，RNG 包含
  `cpu / numpy / random / cuda`。存在状态不等于已验证整条恢复链。
- `actor/pg_loss=0.6938416503`，裁剪前 `actor/grad_norm=267.0700989`。
- student/teacher entropy 分别为 `0.0676771402 / 0.0532579906`，non-finite 计数为 0。
- loss-input 梯度恒等式误差 `1.1674618e-10`，不是完整模型参数梯度误差。
- mean response length `4630.25`，truncation ratio `0.25`。
- `scalars.jsonl`、`window_steps.jsonl` 和 `step_00001.npz` 均已生成。
- ordered prompt hash：`4765d3ee7d88e4ebb256c82aa37cf4c2cc5695981ff8cfafe97ac505e50332e8`。

上述单步数值仅用于可运行性检查，不表示梯度稳定，也不用于评估方法优劣。
vLLM 内部采样流不在上游 checkpoint 保证范围，不宣称恢复后生成逐位一致。

Step2 恢复证据：

- 日志显示恢复 `global_step_1` 的 model / optim / extra_state 和 window state。
- `window_offset=1`，符合独立 RNG 的第二抽；保存状态中的抽样位置匹配。
- `samples_yielded=8`、`snapshot_step=2`，未重新消费首批 4 个 prompt。
- 实际读取四份 optimizer，所有状态的更新计数为 2；四份 scheduler
  `last_epoch=2`，四份 RNG 仍包含上述四种状态。
- 裁剪前 grad norm `12.4667454`，每步指标、Step1/2 的 NPZ 和完整 checkpoint 均通过审计。
- 退出时的 `atexit TemporaryDirectory.cleanup` 中出现 DataLoader worker
  被终止的警告。发生在 Step2 保存与完成之后，训练退出码为 0；作为收尾警告
  保留，不隐藏或解释成训练中的 OOM，也不因此丢弃该 run。

## Sliding3 恢复检查

目录：`$ROOT/runs/20260911v2r1_sliding_window_probe_seed21_ml2/sliding3`。

Step1 进程 PID `3073278` 于 `2026-09-11T10:24:29Z` 正常退出 0，
首步旧审计及 window 审计均通过。恢复进程 PID `3084980` 于
`2026-09-11T12:18:41Z` 正常退出 0，Step1/2 的旧审计和 window 审计均通过。

- 两组的 `artifact_hashes.sha256` 逐字节一致，模型、tokenizer、训练与评测数据身份一致。
- 首步 ordered prompt hash 与 prompt multiset hash 均与 Random3 相同。
- `actor/pg_loss=0.6955979235`，裁剪前 `actor/grad_norm=262.4262695`。
- loss-input 梯度恒等式误差 `1.1674618e-10`，所有已记录 non-finite 计数为 0。
- mean response length `4630.25`，truncation ratio `0.25`。
- window state 为 `mode=sliding, last_step=1, current_offset=0, rng_state=null`；
  `current_offset=0` 只是占位标记，方法实际平均三个 phase，不是仅训练 phase0。
- 实际读取四份 optimizer 与 extra_state：均为更新计数 1、scheduler epoch1，
  四种 RNG 状态齐全；数据 checkpoint 的 `samples_yielded=4`。
- 退出时 `atexit dump_compile_times` 同样出现 DataLoader worker 被终止警告。
  记录该收尾信息；首步保存完成、退出码为 0，不据此补造训练中的 OOM。

Step2 的检查已通过：

- 实际读取四份 optimizer / scheduler / RNG，计数均推进到 2，数据游标为 8。
- window state 为 `mode=sliding, last_step=2, current_offset=0, rng_state=null`。
- 第 1、2 步的 ordered prompt hash 和 prompt multiset hash 均与 Random3 相同。
- 第二步 ordered prompt hash：`bfd467fe2b9692f9caa18058ccf484ac6383823de379e08440725d1dc37856ea`。
- 裁剪前 grad norm `39.4018059`。该值与 Random3 的差异不是方法效果或
  参数梯度方差的结论，两组更新后生成内容允许分叉。
- 退出析构 `MathMultiProcessEnv.__del__` 中也有 worker 被终止的收尾警告；
  完整 checkpoint、每步记录和退出码均通过审计。
- 两方法的实际诊断已生成同色标位置热图、phase 输入导数曲线和三个标量图，
  位于 probe 根目录的 `gate_paired_heatmaps/`，仅用作绘图验收。

## 正式队列已启动

状态快照：`2026-09-11T12:25:30Z`。

- 根目录：`$ROOT/runs/20260911v2r1_sliding_window_seed21_ml2`。
- 队列 PID：`3094013`，服务器端 `nohup`，已确认父进程为 PID1。
- Random3 训练 PID：`3094023`，启动时正在初始化；Sliding3 尚未开始，已排队。
- 两组 run card 只在 variant、window mode、experiment name 和输出目录上不同。
  模型、数据、训练/采样/评测参数相同，artifact hashes 逐字节一致。
- 启动前再次核对两个门禁审计通过、四卡空闲、历史 grader SHA256 一致。
- 队列依次训练两组各 200 steps，再完成两组 Step50/100/200 的四任务 n=8
  全量评测、历史 grader 重评分和探索性配对比较；不启动其他方法或 seed。
- 保存所有规定 checkpoint、监控、热图和失败产物；队列遇到失败时停止并
  留存证据，不自动换参数、重跑失败任务或删除 checkpoint。

状态查询以 ml2 的 `queue_state.json`、`queue.log`、各 run 的 `train.pid`、
`logs/nohup.log`、`diagnostics/window_steps.jsonl` 和退出码为准，不能把本文快照
当成持续更新的实时状态。后续文档提交不改变冻结 runtime，控制命令需显式固定版本：

```bash
SOURCE_COMMIT=fbad852a638de18e20d571a061b2be0437942a38 \
  bash scripts/sliding_window_ml2_control.sh random3 status
```

该快照下没有完成的正式评测，不声称 Sliding3 有效或优于 Random3。
门禁数据只用于验收；名义 rollout 数不代表独立样本数，单 training seed 不
支持跨 seed 稳健性结论。
