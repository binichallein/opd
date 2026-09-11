# 滑动窗口 OPD：边界稳健性与有效性验证方案

状态：设计完成；窗口代码尚未实现，本方案没有启动新训练。
日期：2026-09-11。运行机器限定为 `ml2`，不连接或依赖 `train`。

## 1. 研究问题与结论范围

核心假设：固定 block3 的起点会人为切断一部分短期依赖；平均三个分块
偏移所对应的完整 loss，可以减轻边界依赖，并可能改善最终任务表现。

本实验分开回答三个问题：

1. 实现是否满足 `sliding3 = mean(offset0, offset1, offset2)` 的 loss/梯度
   恒等式？这是正确性检查，不是方法有效性的证据。
2. 真实模型、真实 rollout 上，分块偏移造成的梯度差异是否足够大？
3. 消除偏移选择的随机性，是否改善固定训练预算下的准确率和稳定性？

理论只保证：对同一 checkpoint 和同一 rollout，完整偏移平均去除了随机
选择偏移的方差分量。它不保证优于某个固定偏移，不保证更接近完整序列
KL 梯度，也不保证梯度 clipping/Adam 更新或最终正确率同步改善。
不把 sign-flip 或跨 token 反馈自动标记为错误归因。

## 2. 当前可用资源与实验谱系

2026-09-11 只读检查确认：

- ml2 有 4 张 NVIDIA A100-SXM4-80GB，当时均为 0% GPU 利用率。
- 内存约 935 GiB；实验存储路径可用，NAS 展示的容量不等于项目配额。
- 下列模型、训练/评测文件与旧 token/block3 的 Step50/100/200 目录存在。
- 训练文件、评测文件及历史 grader 的 SHA-256 已重新计算，与旧记录一致。
- 模型目录存在不等于全部权重已重新验证；训练前重新核验权重、tokenizer
  与历史不可变 manifest。历史 Qwen Hub revision 未完整记录，不补造 revision。
- 远端旧项目目录不是 Git checkout；执行时建立新的、由本地 Git commit
  导出的不可变 runtime，记录代码清单哈希，不原地修改旧实验目录。

记远端项目根目录为：

```text
ROOT=/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd
```

| 项目 | 固定值 |
|---|---|
| Student | Qwen3-1.7B-Base，与历史 ml2 对照相同 |
| Student 路径 | `/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/models/Qwen3-1.7B-Base` |
| Teacher | Qwen3-4B-Base-GRPO，与历史 ml2 对照相同 |
| Teacher 路径 | `$ROOT/models/Qwen3-4B-Base-GRPO` |
| 数据目录 | `$ROOT/data/math_opd_dapo17k_hf_full_eval4` |
| 环境候选 | `/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl`，启动前重新验收 |
| 上游代码 | `external/revisiting_opd`，固定 commit `f32f284f25bae5b16d2d44ee336b52851dccc736` 加本仓库补丁 |
| 新运行目录 | `$ROOT/runs/20260911v1_sliding3_seed{seed}_ml2/{variant}` |

当前主实验只有这对 Qwen 师生。DeepSeek/JustRL 跨师生验证作为后续独立
实验，届时在 ml2 从公开来源准备模型，不能假设可以从 train 复制。
即使本轮成功，也只声明该模型对上的多 seed 证据。

## 3. 四组方法与唯一变量

| ID | 方法 | 分块方式 | 作用 |
|---|---|---|---|
| `token_opd` | sampled-token OPD | k=1 | 原始外部代码基线 |
| `fixed3` | 原有 block3_mean | k=3，偏移恒为 0 | 主比较对象 |
| `random3` | 随机偏移 block3_mean | 每个 optimizer step 均匀抽取 r=0/1/2 | 区分随机化边界与完整偏移平均 |
| `sliding3` | 步长 1 的滑动窗口 | 平均 r=0/1/2 的完整 PPO loss | 待验证方法 |

所有组都从同一个初始 Student 权重独立训练。旧 token/block3 checkpoint
只用于阶段 A 的冻结模型诊断及历史参照，不代替阶段 B 的新对照，不把
旧单 seed 成绩与新 seed 成绩拼接成三个重复实验。

### 3.1 精确定义与归一化

在当前上游的每个 token-mean loss 归约单元内，令 N 为有效 response token
数。外层 micro-batch/梯度累积归约保持上游实现，不额外改为另一种全局
token 加权方案。窗口 W 的有效长度是 m_W，定义：

```text
A_i = stopgrad(teacher_logp_i - old_student_logp_i)
a_W = sum(A_i in W) / m_W
r_W = exp(sum(current_logp_i - old_logp_i in W))
L_r = sum(m_W * dual_clip_PPO_loss(a_W, r_W) for W in partition_r) / N
L_sliding3 = (L_0 + L_1 + L_2) / 3
```

这与当前固定 block3 的 `block_mask=m_W/3` 加 masked mean 等价。
各偏移以每条 response 的第一个 token 为坐标原点；保留首尾不满 3 的
截短窗口。每个有效 token 在每个偏移中恰好出现一次，在全部三个偏移中
出现三次。窗口不能跨 response、EOS、padding 或非连续有效 mask 区域。

例如有效 response 只有 6 个 token，三个偏移明确约定为：

```text
r=0: [1 2 3] [4 5 6]
r=1: [1] [2 3 4] [5 6]
r=2: [1 2] [3 4 5] [6]
```

对每条连续有效区域重新采用上述局部坐标；不能让上一条 response 的长度
决定下一条 response 的分块起点。

禁止以下替换：仅枚举完整窗口并丢弃首尾；随意多除或少除 3；对三个偏移
分别进行 gradient clipping/Adam step 后平均参数；先平均 advantage 再用
token ratio clipping。它们均不等于本次预注册的滑动窗口方法。

`sliding3` 合成一个标量 loss、累积一次梯度、进行一次 optimizer step。
复用同一组 old/current/teacher log-prob，窗口量用前缀和或三个偏移视图
计算，不重复做三次教师前向。增加的是标量窗口计算，不预先宣称加速。

### 3.2 随机偏移的可复现性

`random3` 在一个 optimizer step 内，所有 rank、micro-batch 和 trajectory
共用同一 r，并在该 step 内保持不变。r 从独立的 NumPy PCG64 RNG 采样，
`offset_seed = 910000 + training_seed`，不消耗数据或 rollout RNG。
每步记录 r；checkpoint 保存 offset RNG 状态及下一步位置。恢复不得多抽
一次或重复使用上一步偏移。不强制三步轮转，因为那是另一个对照方法。

## 4. 阶段 A：冻结模型上的机制检查

### 4.1 固定样本与模型锚点

- 锚点一：原始 Qwen3-1.7B-Base。
- 锚点二：历史 ml2 token OPD 的 Step100，路径为
  `$ROOT/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/checkpoints/global_step_100`。
- 先审计 checkpoint 内容和加载结果，不能仅凭目录存在认定 checkpoint 可用。
- 从相同 DAPO 训练池确定性选取 64 个不同 prompt，probe selection seed=9211。
  用规范化 prompt hash 去重，并排除本轮三个 training seed 的正式训练
  prompt schedule 中出现的题目，以及与四个评测集规范化 prompt hash
  相同的题目。冻结 probe manifest 后不按结果换题。此检查不等同于已排除
  所有语义近重复，也不改动历史训练池。
- 每个锚点为每题生成 8 个回答，seed=9201..9208，temperature=1.0、
  top-p=0.9、最大 response=16,384。两个锚点总共 1,024 条 rollout。
- 每锚点按 4 prompt x 8 rollout 组成 16 个 batch。每个 batch 的四种
  方法严格使用相同 token 序列、mask、teacher log-prob 和 old log-prob。

### 4.2 数学与实现门禁

1. 小词表枚举检验 k=1、长度 1/2/3/4/5、变长 batch、EOS 和 padding。
2. `fixed3` 偏移 0 的 loss、gradient 与现有实现一致；token 分支保持一致。
3. 显式窗口版与三个偏移平均版，在 clipping 未激活及已激活的合成样本上
   都满足 loss/gradient 恒等式，覆盖正负 advantage 与 dual clipping。
4. FP64 小例子容差 `rtol=1e-10, atol=1e-12`；FP32 单测容差
   `rtol=1e-5, atol=1e-6`。
5. 模型梯度在 FP32 中归约统计；BF16 反向可能改变累加顺序，真实模型的
   梯度恒等式相对误差门槛为 1%。超出则定位精度或实现问题，不能直接
   放宽门槛宣布通过；近零梯度额外报告绝对误差。
6. 统计以裁剪前、尚未 optimizer step 的完整训练参数梯度为准。FSDP
   分片按唯一参数正确累计，不能重复计数。完整梯度可暂存 CPU，不入 Git。

### 4.3 测量量与可证伪结果

每个 batch 计算三个偏移梯度 G_0/G_1/G_2 及其平均 G_bar：

```text
D_phase = mean_batch(mean_r(||G_r - G_bar||^2))
V_batch = mean_batch(||G_bar - mean_batch(G_bar)||^2)
phase_fraction = D_phase / (D_phase + V_batch)
```

这是有限 probe 集上的经验方差分解，batch 不是无限总体，也不是真实
完整序列 KL 的 oracle。分两个锚点报告结果，不只展示差异更大的锚点。
额外报告梯度 cosine 矩阵、各偏移 norm、逐层 D_phase、相对响应位置及
`position mod 3` 的信用分配差异。旧序列结尾不足 3 的窗口单独分层展示。

若偏移差异接近数值误差，则“边界选择噪声是当前重要瓶颈”缺乏支持。
若 D_phase 明显存在，也只能证明有可被平均掉的分量；不能把恒等式带来的
必然差值当作性能提升的证据。阶段 A 不使用四个 benchmark 的成绩调参，
不按哪个偏移更好挑选固定基线。阶段 B 不以阶段 A 是否有利作为择优门槛。

## 5. 阶段 B：三 seed 的配对训练

Training seeds 固定为 21、22、23，每个 seed 都从原始 Student 开始训练
四种方法，共 12 个 run。先完成 seed21 四组作为执行批次，再完成 seed22/23；
除工程或资源阻塞外，不因首个 seed 的成绩有利或不利而增删后续 seed。

| 参数 | 固定值 |
|---|---|
| 数据 | DAPO-Math-17K raw 1,791,700-row pool；不去重替换训练池 |
| steps | 200；不是完整一个数据 epoch |
| 每步输入 | 4 prompt，每 prompt 8 rollout，总计 32 条 trajectory |
| 优化 | LR=2e-6；ppo_mini_batch_size=32；ppo_epochs=1；沿用固定优化器配置 |
| OPD | sampled log-ratio；opd gamma=0；outcome reward weight=0；entropy coefficient=0 |
| PPO | vanilla/dual clip；clip low/high=0.2；dual clip c=3；grad_clip=1.0 |
| 长度 | max prompt=2,048；max response=16,384；prompt truncation=middle |
| 采样 | temperature=1.0；top-p=0.9；data/rollout/environment seed 同 training seed |
| 数据过滤 | filter_overlong_prompts=false，与历史 ml2 run 一致 |
| 资源 | 4 x A100 80GB；actor/reference/rollout-logprob micro-batch=1/1/4 |
| vLLM | gpu_memory_utilization=0.6；max_num_batched_tokens=18,432 |
| 内存 | gradient checkpointing；actor parameter/optimizer offload |
| 保存 | Step50/100/200 完整 checkpoint；保留全部；不自动删除 |

每个 seed 的四组使用同一 prompt schedule，并在每一步记录
`prompt_batch_sha256` 和 prompt/rollout ID。模型更新不同导致 rollout 内容
不同，这是正常的 on-policy 分叉，不强制跨方法复用旧训练 rollout。

为分散顺序效应，执行顺序固定为：seed21 token/fixed/random/sliding；
seed22 sliding/token/fixed/random；seed23 random/sliding/token/fixed。
顺序写入计划，不按中途得分改变。所有 run 独占四卡，训练和 full eval 顺序调度。

### 5.1 监控与图表

- 每步：PG loss、裁剪前 grad norm、长度、截断率、ratio clipping、耗时、
  non-finite 计数，以及训练 prompt batch hash。
- Step1 及每 5 步：student/teacher entropy、signed/absolute gap、Top-16
  overlap、双方 overlap mass、overlap-token advantage。
- 保留 sign-flip、weighted sign-flip 和 normalized leakage 的历史定义，
  同时标注窗口方法下的定义变化。滑动窗口的未裁剪有效 token coefficient
  由各覆盖窗口正确累加得到；不能直接套用只支持 disjoint block 的旧函数。
- 单独记录 PPO clipping 后的 token loss derivative、窗口有效长度、覆盖次数
  以及 token 被至少一个 clipped 窗口覆盖的比例，不能只报窗口数量比例。
- 每 25 步在当前 rollout 的首个 prompt group 上计算三个偏移的冻结梯度
  诊断；各方法同频率执行。诊断不更新权重、优化器或 RNG，额外开销单列。
- 图表：step x token-position 热力图、position-mod-3 对比、偏移梯度 cosine
  和 D_phase 曲线、四方法均值/单 seed 曲线、clip/长度/梯度及 wall-time 对比。
- 位置热图固定 128-token bins；每个 bin 保存样本数，低覆盖区域标记缺失，
  不用 0 伪装为低熵。不同方法共用色标。

### 5.2 运行门禁与失败处理

所有方法分别通过 Step1 保存并恢复至 Step2 的 resume gate。随机偏移 RNG、
所有 rank RNG、sampler、optimizer、scheduler 和模型状态均须正确恢复。
训练前重新核验所有模型、数据、grader、依赖版本与不可变 runtime commit。

运行通过 detached 进程执行。OOM、non-finite、产物缺失或数据/hash 不一致
时停止并保留失败目录；不能静默调学习率、长度、batch 或 clipping。若必须
改变数值相关配置，则新建协议版本并重建可比对照。
有限但准确率低或出现高熵的 run 属于实验结果，不能换 seed 重跑取代它。
无法完成的 run 必须列为失败/缺失，不能只对幸存 run 报平均性能。

## 6. 完整评测与统计

### 6.1 固定评测

- 数据：MATH500 500、AIME24 30、AIME25 30、AMC23 83，共 643 题。
- 每个 run 的 Step50/100/200 全量评测；每题 n=8，eval seed=21..28。
- temperature=1.0、top-p=0.9、max response=16,384、thinking 开关关闭。
- 不把 eval 的八个 seed 当成八次训练。原始 Base 也按同一配置评测一次。
- 主评分器使用历史 `historical_utils_sha04f7.py`；保留内置 grader 的原始
  成绩作为敏感性视图。禁止只选有利 grader 或有利 checkpoint。
- 配对键为 `(task, example_id, rollout_id, eval_seed)`，四组必须完整一致。
- Avg@8 为每题八次的平均正确率；Pass@8 为至少一次正确的题目比例；
  macro 为四任务等权平均，同时保留每任务成绩和长度/格式错误。

### 6.2 主比较与结果分类

主比较是 Step200 的 `sliding3 - fixed3`，co-primary 为 macro Avg@8 和
macro Pass@8。每个 training seed 先独立计算差值，再对三个 seed 等权平均。
报告三个差值、平均差值、seed 间 SD、以及配对差值的 95% t 区间（df=2）。
这依赖近似正态假设，只有三个 seed 时区间可能很宽，不能作强泛化承诺。

额外报告每 seed 的 10,000 次 task-stratified paired prompt bootstrap
（seed=20260911），作为固定 checkpoint 下的评测题目不确定性。不能把
三个训练 seed 的 24 个回答当作独立训练重复，也不把题目 bootstrap CI
当作覆盖训练方差的 CI。

- 强性能支持：两个主终点在三个 seed 的点估计均正，且各自 seed 级 95% t
  区间下界均大于 0；同时完整报告安全性/失败情况。
- 方向支持但不充分：两个主终点的跨 seed 均值均正，但未达到以上条件。
- 未支持一致收益：至少一个主终点的跨 seed 均值非正；不等同于证明普遍无效。
- `random3-fixed3`、`sliding3-random3`、`sliding3-token_opd`、Step50/100 和
  分任务结果是预先指定的次要分析，报告效果及探索性 CI，不将其中某个
  偶然显著结果替换主比较。不能仅凭胜过 fixed3 就宣称超过 token OPD。

机制解释要求结合阶段 A 和在线诊断：若有性能提升但 phase 差异很小，不能
归因于边界噪声；若 phase 分量被平均掉但性能没提高，则只支持估计器性质。
若随机偏移已取得相近效果，需报告额外做完整平均的收益与计算成本。
这些观测支持或削弱机制解释，但不构成唯一因果机制的证明。

## 7. 数据、产物和工作量

```text
train_sha256 = cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf
eval_sha256 = c5c9f1b3f65eb40053cae14c005c39e29443d04acbbabba873e98d64818a9680
grader_sha256 = 04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f
```

- 阶段 A：2 个锚点 x 64 prompt x 8 rollout = 1,024 条诊断 rollout。
- 阶段 B：4 方法 x 3 seed = 12 run，共 2,400 training steps，76,800 条
  nominal training rollout；不是 76,800 道独立题目。
- 正式 eval：12 x 3 checkpoint x 643 x 8 = 185,184 条；加 Base 的 5,144 条，
  共 190,328 条。resume gate 与失败重试的消耗单列，不隐藏在正式预算里。
- 训练、teacher scoring、backward/update、额外诊断和 eval 时间分别统计。
  用本轮成功 gate 及前 10 步估算实际排期，不用 A800 旧耗时承诺 A100 完成时间。
- 每个 run 保存 resolved config、命令、模型/数据/代码哈希、依赖版本、PID、
  RNG 与完整 checkpoints；原始 rollout/梯度/日志留在 ml2，不入 Git。
- 仓库只保存协议、配置、curated JSON、图和 HTML，失败结果也需记录。

## 8. 执行前的实现工作

以下尚未实现，不应把配置摘要当作可直接启动的 Hydra 配置：

1. 在 `opd_ext/window_supervision.py` 实现偏移分块与三偏移完整 loss 平均，
   配合 `tests/test_window_supervision.py` 验证权重、mask、梯度和 clipping。
2. 修改 `patches/revisiting_opd/blockwise_sampled_opd.patch` 的 actor/driver
   传参，使一个 optimizer step 的 offset 在所有 rank/micro-batch 固定；
   保持 k=1 和 fixed offset0 的默认行为。
3. 给 `opd_ext/diagnostics.py` 增加窗口覆盖及有效 token coefficient，
   增加 checkpoint/RNG 与 freeze-probe 的无副作用测试。
4. 增加独立 ml2 launcher 和冻结 probe 脚本，显式覆盖旧入口中的 train
   默认路径；生成不可变 runtime 及本方案的 resolved run cards。
5. 复用 `scripts/eval_qwen3_math_vllm.py` 的完整评测与审计；扩展配对统计
   支持四种 variant 和 seed 级重复，不修改旧比较结果。
6. 通过本方案阶段 A、所有方法的 resume gate 和已有相关回归测试后，
   才执行阶段 B。当前交付止于方案设计与归档。
