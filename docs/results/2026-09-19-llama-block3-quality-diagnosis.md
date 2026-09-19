# Llama 历史提示对照：Token 结果与 Block3 输出退化

## 范围与状态

- 本文记录 2026-09-19 上午的现场监督，不替代完整的 Block3 benchmark 评测。
- 仅访问 ml2；未访问 train，未更换数据、模型、提示、seed、学习率或 loss。
- Run：`runs/20260918v4_llama32_historical17_recovery_seed21_ml2`。
- 远端根目录：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
- 训练/评测部署保持 `94be7ea1d659309256c8356681925bb9710895c4`，恢复控制器保持
  `b00ad93385ef367cc88c499d3c89d9feaa515327`。文档提交不是运行时版本。
- 学生/教师是 ModelScope 的 Llama-3.2-1B-Instruct / Llama-3.2-3B-Instruct。
  本文的“初始模型”指未经本次 OPD 的 Instruct 模型，不是预训练 Base 模型。
- 按用户明确要求复刻旧 1.7B 提示差异：训练要求 think 标签，评测不要求；均用 Llama 原生模板。
- 10:44:55 北京时间：Token 已完成 200 步和 50/100/200 三次完整评测；
  Block3 完成 189 步，PID815640，50/100 权重保留，尚未开始正式 benchmark 评测。
  后续进度必须重新读取远端 `queue_state.json`，不能把此快照当实时状态。

### 11:03 更新：Block3 训练完成，进入完整评测

- 11:01:44 北京时间正式训练退出，exit code0；200 步全部完成。
- `acceptance.json` 为 passed，issues/warnings 均为空；完整50/100/200权重及41个诊断点保留。
- `rollout_acceptance.json` 为 passed；200 个逐步归档、6400条名义训练轨迹保留。
  本次全程实际 length-stop 为664/6400=10.375%；Token为952/6400=14.875%。
  Block3 截断较低却存在严重无关输出，不能将这个差值解释为方法成功。
- Step200：平均生成长度11013，grad norm0.781，学生熵9.779、教师熵9.744、
  sign-flip42.54%、normalized leakage0.969；非有限计数均为0。
- 实际 CPU 读取四 rank 的优化器和 extra state：所有 optimizer step200、
  scheduler last_epoch200、LR2e-6，CPU/CUDA/NumPy/Python RNG状态均存在。
  四 rank 的 model 文件均存在且非空；未在这次 CPU 核验中重新加载每个 model tensor。
  `data.pt` 快照200、sampler累计800个prompt、`latest_checkpointed_iteration.txt` 为200。
- 补查189-200步，两组各384条原始记录的 prompt/顺序/采样/协议仍全部一致，
  连同前188步核对覆盖全部200步。
- 两组都在最后清理阶段出现 `MathMultiProcessEnv.__del__` / DataLoader worker killed
  traceback；均已完成最终保存、退出码0，后续审核通过。这是退出清理告警，
  不应写成 Step28 输出退化的触发原因，也没有因此重启训练。
- 11:02:44 开始合并 Block3 Step200，merge exit0。
- 11:03:39 完整评测启动，PID961434；11:04日志已确认加载正确的 Step200 权重。
  Token200/Block3_200的 eval card 只有 model 与 role 不同：temperature1、top-p0.9、
  max tokens16384、每题8次、rollout seeds21-28、external历史grader、相同原生EOS。
  后续按 Step100、Step50继续；当前尚无 Block3 完整 benchmark 分数。
- 11:06进一步确认四个评测worker均完成加载，四张GPU正在执行MATH500生成，
  利用率约77%-80%；尚未发现评测 traceback/OOM。归档文件已建立，最早一次检查仍为空，
  不能把空文件视为生成完成或完整轨迹留存验收。最终需核对实际行数及 acceptance。
- 最终图表由原队列生成，保存在 `block3_mean/figures/`，并复制至
  `/home/tyf/paper/outputs/llama-historical17-supervision/20260919_block3_step200/diagnostics.html`。
  最终位置热力图已实际查看。

## 已完成的 Token 完整评测

全部使用历史 grader，每次包含 MATH500 500 题、AIME24 30 题、AIME25 30 题、
AMC23 83 题，每题 8 个输出，共 5144 个输出。三次 `acceptance.json` 均为 passed。
各 benchmark 独立计分，下表数值均为百分数。

| Benchmark | 初始 Avg@8 | Token50 Avg@8 | Token100 Avg@8 | Token200 Avg@8 |
|---|---:|---:|---:|---:|
| MATH500 | 14.00 | 14.15 | 14.90 | 15.55 |
| AIME24 | 0.42 | 0.00 | 0.42 | 0.42 |
| AIME25 | 0.00 | 0.00 | 0.00 | 0.00 |
| AMC23 | 5.27 | 6.78 | 6.48 | 6.02 |

| Benchmark | 初始 Pass@8 | Token50 Pass@8 | Token100 Pass@8 | Token200 Pass@8 |
|---|---:|---:|---:|---:|
| MATH500 | 46.40 | 44.80 | 45.80 | 45.20 |
| AIME24 | 3.33 | 0.00 | 3.33 | 3.33 |
| AIME25 | 0.00 | 0.00 | 0.00 | 0.00 |
| AMC23 | 28.92 | 32.53 | 31.33 | 28.92 |

Token200 的 MATH500 Avg@8 比初始高 1.55 个百分点，但 Pass@8 低 1.20 个百分点；
AMC23 的 Avg@8 高约 0.75 个百分点，但不及 Token50。AIME 未见提升。
这些是单 seed 观察，不是统计显著性结论，也不能写成所有任务一致改善。

| Benchmark | 初始 format error | Token200 format error | 初始实际截断 | Token200 实际截断 |
|---|---:|---:|---:|---:|
| MATH500 | 42.28 | 24.43 | 1.38 | 7.53 |
| AIME24 | 32.50 | 32.92 | 2.92 | 16.67 |
| AIME25 | 29.17 | 30.00 | 3.75 | 21.25 |
| AMC23 | 34.94 | 22.44 | 1.36 | 12.95 |

格式指标沿用当前评测器的既定定义；格式改善不等于数学推理正确，也不能代替截断率。
证据位置：本 run 的 `reused_initial_evaluation.json` 和
`evaluations/token_opd_step{50,100,200}/acceptance.json`。
历史 grader SHA256：`04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。

## 对照条件与数据完整性

1. 两组 `run_card.json` 逐字段比较：只有 variant、实验/输出目录名、
   `opd_block_size`（1/3）与 `opd_block_advantage_mode`（sum/mean）不同；
   size=1 时 sum/mean 聚合分支直接旁路。
2. 两组 `data_manifest.json` 字节一致；同一训练文件、seed21、4 个 prompt/步、
   每 prompt 名义 8 个 rollout、LR2e-6、200 步、max response16384、microbatch1、vLLM0.6。
3. 对前 188 步逐条读取两组各 6016 条原始轨迹，比较 sample index、source extra info、
   prompt token IDs、sampling、protocol、source commit、EOS：没有不匹配步骤。
4. 首步 32 个实际 response token 序列逐条完全一致，之后才随训练分化。
5. 两组记录的 27 个脚本文件重新计算 SHA256，全部匹配运行时清单；
   冻结部署 `.expected.sha256` 的1501项完整校验也通过，无缺失/不匹配。
6. 已知历史限制仍在：同一个 prompt 的 8 个训练副本使用相同请求 seed，
   已核对的输出在组内重复。这不是 32 个独立样本/步，不能把名义 rollout 数当独立样本数。
   未在单臂中途修改采样；独立 n=8 评测是另一条生成路径，不能混同。

## Block3 退化证据

### 时间顺序

| Step | 学生平均熵 | 教师平均熵 | 符号翻转率 | 学生 overlap mass |
|---|---:|---:|---:|---:|
| 1 | 0.512 | 0.570 | 22.78% | 98.04% |
| 20 | 1.061 | 0.956 | 28.42% | 95.49% |
| 25 | 1.665 | 1.444 | 28.66% | 89.16% |
| 30 | 5.862 | 5.725 | 29.63% | 45.79% |
| 35 | 9.237 | 9.162 | 31.19% | 9.40% |
| 50 | 9.626 | 9.564 | 35.10% | 4.91% |
| 100 | 9.590 | 9.527 | 42.46% | 5.17% |
| 185 | 9.662 | 9.604 | 44.16% | 4.29% |

- 抽查原始回答：Step28 / data index1463489 已出现无关词语串，长度6368，
  引擎以 stop 结束，尾128 token 的平均 rollout logprob 为 -9.738。
  这是“至少在 Step28 已出现”的证据，不声称人工穷尽确认了首次发生时间。
- Step35 / indices891149、573962、1255372 等出现长段无关词语；
  Step50 同时有正常 stop 与 length 截断的退化回答。
- Step160/180/183 等不同问题出现相近无关尾段，多条总长度11013并以原生 EOS 结束。
  因此“没有达到16384上限”不等于回答恢复正常。
- Block3 最大裁剪前 grad norm 为161.621，发生在 Step75。
  高熵和乱码更早已出现，不能把这个晚发生的尖峰直接认定为最初触发原因。
- 10:44 快照中 Block3 未发现 OOM、NaN/Inf scalar 或运行异常 traceback；
  但数值有限不能否认输出行为已明显退化。
- Token 也存在重复、跑题、高熵个别 batch，不能把 Token 写成完全健康的基线。
  对比的是各自 on-policy 轨迹，并非在相同生成前缀上评估两个学生。

### 位置统计与读图边界

抽查两组 Step20/25/30/35/50/180，NPZ 的学生熵有效计数之和均等于对应原始
rollout 的实际 token 总数。这支持统计没有简单地把 padding 作为有效 token，
但不是对所有前向计算和样本重排逻辑的穷尽证明。

Block3 的位置分段熵：

| Step | 前512 token | 512-2048 | 2048-4096 | 4096-8192 |
|---|---:|---:|---:|---:|
| 25 | 1.679 | 1.301 | 无观测 | 无观测 |
| 30 | 3.552 | 8.198 | 5.160 | 无观测 |
| 35 | 4.196 | 9.044 | 10.016 | 10.001 |
| 50 | 5.296 | 9.827 | 10.025 | 9.875 |
| 180 | 6.769 | 10.061 | 9.999 | 9.984 |

熵在前段也上升，不只存在于16K末尾。各步题目/存活长度不同，不能据此直接证明
同一个推理过程的错误从后向前因果传播。

Step185 Top-16 overlap ratio 为81.34%，但共享 token 仅承载学生4.29%、教师4.65%的
概率质量。高 top-k 集合重合与小熵差可以同时出现在双方都很不确定的学生前缀上，
不能单独解释为高质量能力对齐。教师熵测量是在学生前缀上，而非教师独立生成的轨迹。

Step185 符号翻转率44.16%，按原始 advantage 绝对值加权后为37.26%。
归一化 leakage 为0.973，定义是 `sum(abs(block_mean - token_advantage)) / sum(abs(token_advantage))`。
它不是“97.3%的 token 都错了”，也不是梯度范数；不能把相关性直接写成 collapse 的因果证明。

## 实现复核：这不是只有 advantage 共享的单因素消融

独立只读复核及本地 CPU 小例子确认，当前执行的是历史约定的完整 block objective，
不是把 block mean 广播后仍使用原 token PPO loss：

1. block 内 old/current logprob 求和，因此 PPO ratio 是 token ratio 的乘积。
2. block mask 为 `有效长度 / k`；`token-mean` reduction 接收的已是 block 数组，
   分母是 block mask 之和。这里“token-mean”这个配置名不等于原始 token 数分母。
3. 在一个 response 有 N 个有效 token、一个 block 实际含 n 个 token、ratio=1 时，
   对当前 token logprob 的导数为 `-n * block_mean / N`。
   而“广播相同 block_mean + 原 token loss”是 `-block_mean / N`。
   完整 block3 相差3倍；尾部短 block 按实际长度而不是固定3倍。

调用实际 `compute_policy_loss` 的 CPU 核对（无训练、无新权重；测试依赖使用既有测试夹具）：

| 输入条件 | Token loss | Block3 mean loss | Token 各 logprob 梯度 | Block3 各 logprob 梯度 |
|---|---:|---:|---|---|
| 3 token，advantage全1，ratio全1 | -1.0 | -1.0 | [-1/3, -1/3, -1/3] | [-1, -1, -1] |
| 同上，ratio全1.1，clip上界1.2 | -1.1 | -1.2 | [-0.3667, -0.3667, -0.3667] | [0, 0, 0] |

第二行中 block ratio 是1.331，触发裁剪；每个 token 的1.1未触发。
这个合成例子只证明实现差异，**不是本次训练触发了相同裁剪情形的证据**。

实现位置：`external/revisiting_opd/verl/trainer/ppo/core_algos.py` 的
`aggregate_blockwise_policy_inputs`、`compute_policy_loss`、`agg_loss`。
该定义已在 `docs/plans/2026-07-11-ml2-block3-replication-design.md` 中约定；
除以3、改用逐 token ratio 均属于新方法版本，不能冒充恢复既有实验的工程修复。

边界与含义：

- 这里推导的是 loss 对 logprob 输入的梯度，不是 Adam 最终参数更新。实际还有梯度裁剪、
  优化器状态、不同 advantage 方向和参数 Jacobian。运行日志确认两组 grad_clip 均为1。
- microbatch1 下，每条 response 内先归一化，再累积/平均，不是整个 batch 按总 token 数统一归一化。
- 符号翻转与 leakage 诊断描述的是 block mean 对 token advantage 的替换，
  不包含上述长度乘子，不能拿 leakage 直接当最终梯度改变量。
- 当前配置单 minibatch、单 PPO epoch，loss 中的 ratio 名义上接近1；
  post-update drift 只是更新之后的诊断，不会追溯性地裁剪已经执行的更新。
- rollout 使用 top-p0.9，entropy/logprob scoring 使用温度缩放的全词表分布；
  这是两组共享的既定估计器条件，不是本次发现的单臂配置差异。
- 因而这组对照测试的是“历史 Block3 Mean 方法整体 vs Token”，不能把差异完全归因于
  advantage 平滑或符号翻转。未发现足以确认造成此次 collapse 的实现错误；
  归因需要后续专门控制梯度尺度、ratio 粒度等因素的实验，当前未启动。

## 核验、图表与后续

- 本次没有改训练或评测代码，没有中断、重试或另开 GPU 实验。
- 本地聚焦测试：`test_revisiting_opd_block_policy_loss.py`、`test_opd_diagnostics.py`、
  `test_llama32_queue.py`、`test_historical17_queue.py`，42 passed。
  仓库要求的 `test_block_supervision.py` 另有4 passed，`clean_opd_train.py` 编译检查通过。
  单元测试通过不是 GPU 训练正确性的充分证明。
- 使用冻结部署的既有 CPU 绘图脚本生成新快照，不覆盖最终 figures：
  `block3_mean/supervision_snapshots/20260919_1045/`。
- 本地副本：`/home/tyf/paper/outputs/llama-historical17-supervision/20260919_block3_1045/diagnostics.html`。
  已实际查看 `position_heatmaps.png`；灰色表示该位置没有有效观测。
- 原队列已完成 Step200 状态保存、全部 raw rollout 与诊断审核，正在按200/100/50顺序
  执行 Block3 完整评测并保存每个输出。发生工程错误时先保留证据、定位原因，不能默默改算法重跑。
- 在 Block3 三个 checkpoint 的分 benchmark 结果出现之前，不写“该师生对验证成功”，
  也不根据训练 rollout 直接编造评测分数。
