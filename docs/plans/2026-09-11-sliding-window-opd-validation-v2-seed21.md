# 滑动窗口 OPD v2：seed21 双方法范围修订

状态：范围已修订，等待实现验收与门禁；本修订没有新增实验结果。
日期：2026-09-11。协议版本：`20260911v2_seed21_random3_sliding3`。
用户已授权在实现验收与门禁通过后启动本轮两次正式训练及规定的完整评测；
这不是门禁已通过或训练已启动的声明。

## 1. 修订依据与适用范围

- [原始 v1 方案](2026-09-11-sliding-window-opd-validation.md)保持原文不变，
  作为四方法、三 seed 设计的历史记录。本修订覆盖其中冲突的执行范围、
  诊断预算、Base 评测和统计规则，不改变未被覆盖的数学定义、正式训练
  参数、完整评测设置、谱系审计与失败/保留规则。
- [当前配置摘要](../../configs/experiments/revisiting_opd/sliding_window_validation.yaml)
  是科学协议摘要，不是可直接执行的 Hydra 配置。
  [报告对应章节](../../reports/block_opd_experiment_report.html#sliding-window-v2-seed21)
  区分当前范围与历史设计。
- 本协议全部远端工作限定在 `ml2`。绝不连接 `train`，不从该主机复制、
  不在其上运行，也不以它为依赖。其他历史记录中的机器名称不是本轮授权。

## 2. 仅两个正式 run，完成后停止

| 顺序 | 方法 | Training seed | 正式 steps | 名义训练 rollout |
|---|---|---|---|---|
| 1 | `random3` | 21 | 200 | 6,400 |
| 2 | `sliding3` | 21 | 200 | 6,400 |

只运行 seed21 的上述两组，完成各组全部 Step50/100/200 评测后停止并报告。
不因有限成绩有利或不利而改变范围；不自动继续 seeds22/23，不增加新
fixed3/token 对照，不更换低分 seed，也不扩展师生组合。后续扩展需要
另行明确授权和版本化协议；当前两组通过门禁后的启动已获授权。

两组分别从同一 Qwen3-1.7B-Base 初始权重独立训练，使用相同 seed21
prompt schedule。每步记录 `prompt_batch_sha256` 与 prompt/rollout ID。
更新后的 on-policy 回答内容可以分叉，不能强制复用历史训练 rollout。
按表中顺序独占 ml2 四卡，训练和 full eval 顺序调度，不按中途得分改序。

`ROOT=/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
启动器使用以下独立目录，不能覆盖历史实验：

- 正式训练：`$ROOT/runs/20260911v2r1_sliding_window_seed21_ml2/{variant}`。
- 小型 resume gate：`$ROOT/runs/20260911v2r1_sliding_window_probe_seed21_ml2/{variant}`。
- 启动记录：原 `20260911v2` 的 Random3 门禁在 Ray 初始化时因 Unix socket 路径过长退出，未进入模型训练；原日志完整保留。`v2r1` 仅缩短缓存路径并换用独立目录，不改变数值配置。

这里的 probe 目录仅用于小型恢复门禁，不表示启动下文已推迟的完整 Stage A。

## 3. 不变的模型、损失与正式训练设置

学生为 Qwen3-1.7B-Base，教师为 Qwen3-4B-Base-GRPO，使用 v1 中的确切
模型路径。数据仍为 DAPO-Math-17K raw 1,791,700-row pool，不去重或替换。
启动前核验模型权重/tokenizer、数据、grader、依赖和不可变 runtime；
历史 Hub revision 未记录的部分保持未知，不能补造为已知事实。

每组 200 optimizer steps；每步 4 prompt x 8 rollout；LR=2e-6；
PPO mini-batch=32、PPO epochs=1；sampled log-ratio OPD，gamma=0、
outcome reward weight=0、entropy coefficient=0；clip low/high=0.2、
dual clip c=3、grad_clip=1.0。max prompt/response=2,048/16,384，
prompt truncation=middle，temperature=1.0、top-p=0.9，
`filter_overlong_prompts=false`。

保留四张 A100 80GB，actor/reference/rollout-logprob micro-batch=1/1/4，
vLLM memory utilization=0.6、max batched tokens=18,432，
gradient checkpointing 与 actor parameter/optimizer offload。
数据、rollout 和环境的初始 seed 都按 training seed21 设置；相同初始
seed 不等于能够逐位恢复 vLLM 的内部采样流。

窗口定义保持 v1：保留不满 3 的首尾窗口，每段连续有效 response 重置
偏移，不能跨 response/EOS/padding/mask 间断；保留按有效 token 加权的
phase loss 与上游外层 micro-batch 归约。`sliding3 = (L_0 + L_1 + L_2) / 3`
平均三个完整 dual-clipped PPO loss，再进行一次 backward/update，
不是三个独立 optimizer updates；复用 old/current/teacher log-prob。

`random3` 每个 optimizer step 从独立 NumPy PCG64 RNG 均匀抽取一个偏移，
`offset_seed = 910000 + 21 = 910021`。同一步所有 rank、micro-batch 和
trajectory 共用该偏移。保存 offset RNG 与下一步位置；恢复不得多抽、
漏抽或重复抽取，也不消耗数据或 rollout RNG。

## 4. 推迟完整模型梯度探针，保留必要门禁

### 4.1 为节省算力推迟的项目

- 完整冻结 Stage A：两个锚点 x 64 prompt x 8 回答，共 1,024 条诊断
  rollout；本轮不生成，不为该探针加载历史锚点，也不作为正式训练前置条件。
- 原每 25 步的三偏移完整参数额外反向传播、参数梯度 cosine、
  逐层 phase variance 与模型参数 `D_phase/V_batch/phase_fraction` 分析。
- 原真实模型完整参数梯度恒等式及 1% 相对误差门槛一并保留在推迟的设计中，
  不作为本轮已通过的验收结果或当前门禁。
- 普通训练 backward、gradient clipping 和原有训练 grad norm 主监控不变；
  推迟额外模型梯度探针不等于取消正常训练梯度监控。

### 4.2 保留的验收与恢复检查

1. 保留 unit loss/gradient identity：显式窗口与三偏移平均一致，
   k=1 与 fixed offset0 回归一致；覆盖长度 1/2/3/4/5、变长 mask、
   截短窗口、EOS/padding、正负 advantage、clipping 激活/未激活与 dual clip。
   FP64 容差 `rtol=1e-10, atol=1e-12`，FP32 容差
   `rtol=1e-5, atol=1e-6`。这是单测梯度，不是完整模型参数探针。
2. random3 和 sliding3 均在指定真实师生模型上通过 Step1 保存、退出、
   恢复至 Step2 的小型门禁。核对模型、optimizer、scheduler、global step、
   数据身份与 sampler/prompt 位置、offset 状态/下一次抽样，以及所有 rank
   的 torch RNG 状态。仅加载权重不算通过。门禁专用样本/长度限制与成本
   单列；正式训练设置不变，重新从公共 Base 出发，不把门禁 steps 算入
   400 个正式 steps。
3. 明确恢复保证的边界：上游不保存 vLLM 引擎内部采样流，因此不承诺精确
   恢复 vLLM sampling stream，也不宣称恢复后生成的 rollout 与不中断训练
   逐位一致。optimizer、数据、offset 和 all-rank torch RNG 的检查不能
   被解释为已经证明整个生成引擎的逐位可复现。
4. 验证监控与 loss-input 诊断不改变模型/optimizer 状态、不消耗 RNG、
   不把诊断导数累加到正常训练梯度；保存实际验收证据。本修订不声明门禁已通过。

OOM、non-finite、产物缺失或 data/hash/state 不一致时停止并保留失败目录。
不静默调整 LR、长度、batch 或 clipping；数值相关配置必须改动时建立新
协议版本和可比对照。有限低分或高熵本身不是替换 run 的理由；未完成 run
必须明确记为失败/缺失。

## 5. 监控与轻量 loss-input 诊断

每步保留 PG loss、普通训练裁剪前 grad norm、长度/截断率、ratio clipping、
non-finite 计数、耗时和 prompt batch hash。Step1 及每 5 步保留师生 entropy、
signed/absolute gap、Top-16 overlap、双方 overlap mass 和 overlap-token
advantage。保留历史 sign-flip、weighted sign-flip、normalized leakage 定义，
同时明确窗口方法下的定义变化，不能套用仅适合不重叠 block 的公式。

热图保留 step x token-position、128-token bins、bin 样本数、跨方法共享色标
和 position-mod-3；低覆盖区域标缺失，不以 0 表示低熵。记录窗口长度、
覆盖次数和被至少一个 clipped 窗口覆盖的 token 比例。展示两个 seed21
曲线，不伪装为多 seed 均值。

轻量 per-phase 诊断采用如下实际计算口径：

- 每 5 步执行一次，即 Step5/10/.../200；在 CPU 上复用同一个已缓存的
  完整训练 batch，不只选首个 prompt group，不增加 rollout。
- 使用更新前的 log-prob 与 mask，`old = current`；把 detached current
  log-prob 作为独立输入叶节点，计算各 phase loss 对这些输入的导数。
  此时 ratio 从 1 开始，这不是参数更新后 clipping 行为的测量。
- 输出 `loss_input_*` 标量摘要，包括各 phase loss、输入导数 norm、
  cosine 与 dispersion；不输出完整参数梯度或将其称为参数梯度指标。
- 零/近零 norm 导致 cosine 未定义时，JSON 中使用数值 0 占位，并同时
  保存有效性标记/有效计数；聚合时只使用有效项。禁止写入 NaN/Infinity，
  也不能把无效项的 0 占位解释为正交。
- 这是 **loss 输入导数诊断，不是完整参数梯度**。不穿过真实模型额外反向，
  不增加模型/教师前向，不改变训练状态或 RNG；不据此报告模型参数
  `D_phase` 或声称参数梯度方差下降。诊断耗时单列，普通训练 grad norm
  仍是原有参数梯度监控，不被该输入空间统计替代。

## 6. 完整评测与历史参照

两组的 Step50/100/200 均完整评测 MATH500=500、AIME24=30、AIME25=30、
AMC23=83，共 643 题；每题 n=8，eval seeds21..28。保持 temperature=1.0、
top-p=0.9、max response=16,384、thinking 关闭。主评分器保持
`historical_utils_sha04f7.py`，SHA-256 为
`04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
内置 grader 结果保留作敏感性视图，不择优选择 grader 或 checkpoint。

当前两组按 `(task, example_id, rollout_id, eval_seed)` 对齐并核对完整性。
报告各任务 Avg@8、Pass@8、四任务等权 macro，以及长度和格式错误。
历史 fixed3/token 仅作背景，不是本轮新跑的配对对照，不进入当前 paired
bootstrap，也不支持严格的新 sliding3-vs-fixed3/token 有效性比较；
同名 seed 不足以建立新的配对实验。

历史 Base eval 只在 manifest 和 resolved eval config 完全一致时复用：
核对 Base 权重/tokenizer、数据身份/题目 ID、解码/thinking、样本数量/seeds
与 grader。身份未知或不匹配则不复用。不自动启动 Base eval；如另行授权，
其 5,144 条 rollout 单独计费，不混入 checkpoint 评测预算。本修订不声明
已有 Base 评测符合复用条件。

## 7. 单 seed 探索性统计与结论边界

主分析为本轮 Step200 的探索性 `sliding3 - random3`，指标为四任务等权
macro Avg@8 和 Pass@8。报告 seed21 唯一配对差值及两组绝对成绩；
Step50/100 与分任务结果保持探索性，不因中途得分替换 Step200 主终点。

seed21 单独做 10,000 次 task-stratified paired prompt bootstrap，
bootstrap seed=20260911，报告 95% 区间。各任务内按题目重采样，两方法
保持配对，每题八个回答整体保留；每次计算等权任务 macro 差值。
该区间只反映固定 seed21/checkpoint 下的题目不确定性，不覆盖训练方差。

配对 training seed 只有一个（`n=1`），不计算 seed 级 tCI 或 seed 级 SD，
不套用 v1 三 seed 强支持分类。八个 eval seed 不是八次训练；不混入历史
run 伪造重复。当前不作有效性、跨 seed 稳定性或完整参数梯度机制结论；
即使未来探索性题目区间不含 0，也不能据此升级结论。本修订没有新增结果。

## 8. 预算与产物保留

| 项目 | 算式 | 本轮正式预算 |
|---|---|---|
| Run 数 | 2 方法 x 1 training seed | 2 |
| 正式训练 steps | 2 x 200 | 400 |
| 名义训练 rollout | 400 x 4 x 8 | 12,800 |
| 完整 checkpoint eval rollout | 2 x 3 x 643 x 8 | 30,864 |
| 冻结完整 Stage A | 推迟 | 0（原设计 1,024） |
| 新增 Base eval | 不自动授权 | 0（另行授权时增加 5,144） |

当前正式评测总数为 30,864。只有另行授权新增 Base eval 后，评测总数才为
36,008。resume gates、失败重试、teacher scoring、普通 backward/update
和轻量诊断成本分别记录；名义训练 rollout 数不是独立题目数。
不以旧硬件耗时承诺新排期，不补造测量结果。

完整保留每个 Step50/100/200 checkpoint，包括模型/tokenizer、optimizer、
scheduler、sampler、all-rank torch RNG、offset 状态、global step、
resolved config 与谱系；vLLM 内部采样流不在上游 checkpoint 保证范围。
不自动删除或裁剪，失败产物也保留。实际运行保存命令、哈希、依赖版本、
PID/日志路径和评测完整性审计；原始权重/rollout/日志留在 ml2，不进入 Git。
