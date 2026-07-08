# OPD / OPSD 研究 idea 备忘

本文档区分三类证据：

- `verified literature`: 已从论文全文或官方报告中核验的方法/发现。
- `pilot evidence`: 本次 clean-room 机制实验得到的初步证据。
- `hypothesis`: 由文献和实验推出的待验证假设。

所有实验边界：只使用官方公开模型和公开数据，不使用用户自训 checkpoint 或旧项目产物；用户提醒的自训练 Qwen3-MoE checkpoint 被明确排除。

## Idea 1: Reliability-Conditioned OPD

核心问题：标准 OPD 在所有 student-visited token 上等强度对齐教师，但教师在学生前缀上的局部可靠性并不恒定。

方法：用 top-k overlap、采样 token 是否落入教师 top-k、teacher margin、teacher entropy 构造 reliability gate，对 KL/implicit-reward update 做软加权。

为什么有研究价值：

- `Rethinking OPD` 指出 OPD 成功依赖 thinking-pattern overlap，且有效概率质量集中在共享高概率 token。
- `Revisiting OPD` 指出 teacher on student prefix 不可靠是 OPD 失败源。
- 现有 top-k / trajectory filtering 方法多把 local support 当作截断集合，而不是直接控制 update strength。

可行性：

- 不需要改模型结构，只改 loss weighting。
- 官方 Qwen3-0.6B/Qwen3-4B 上 64 条诊断显示 mean overlap 0.619、router keep 0.963，说明该配对有可学习但非完美的局部兼容性。
- 本次 pilot10 中 topk_router 从 base 0.420 / standard 0.360 提到 0.450；pilot50 中 topk_router accuracy 0.440，低于 standard 0.460，但梯度更稳，max grad norm 238 vs standard 496。
- seed13 ablation 显示 topk_router 的 GSM100 降到 0.380，低于 seed13 standard 的 0.430；但 MATH100 为 0.270，高于 seed13 standard 的 0.220。说明 routing 的收益不是稳定 accuracy gain，而是任务/seed 依赖的训练调节。
- strict gate 将 avg keep rate 从约 0.956 降到 0.610、avg grad norm 降到 72.1，但 GSM100/MATH100 分别只有 0.410/0.200，支持“过度过滤会欠学习”。
- full split sanity check 进一步收紧结论：GSM8K 上 base/standard/topk/routed_lam110 分别为 0.41698/0.41774/0.41698/0.41926；MATH500 上分别为 0.160/0.168/0.164/0.160。routing 没有稳定超过 standard。
- 200-step 结果显示长训漂移：standard/topk/outcome_reweight/outcome_topk 在 GSM8K 上为 0.321/0.339/0.334/0.332，在 MATH500 上为 0.118/0.132/0.126/0.122，均低于 base 与 50-step。topk_router 是 200-step 组内最好，说明 reliability gate 有减损作用，但无法阻止漂移。
- 同一 200-step run 的 early-stopping curve 显示退化很早出现：standard 从 GSM8K 0.41774/MATH500 0.168 at step50 跌到 0.35254/0.138 at step100；topk 也跌到 0.35254/0.142。reliability gate 更适合作为 early-stopping / trust-region trigger。
- offline robust regrade 改变绝对分数但不改变同轨迹趋势：GSM8K robust primary 中 standard 从 0.46171 at step50 跌到 0.39955 at step100，MATH500 从 0.202 跌到 0.162。漂移不是简单 boxed/exact grader artifact。
- reference-anchored 100-step 实验给出混合证据：`anchored_topk_router` 的 max grad norm 为 138，低于 standard 201 和 topk 224；GSM8K robust primary 为 0.41926，高于同 run standard 0.40713/topk 0.39424，但仍低于 base 0.46247。MATH500 robust primary 为 0.178，低于 base 0.202。
- explicit reference-penalty 100-step 实验进一步提醒要看 run-to-run variability：该 run 的 standard 在 GSM8K robust primary 达到 0.46702，略高于 base 0.46247，但 MATH500 只有 0.196，低于 base 0.202；`ref_penalty_standard` / `ref_penalty_topk_router` 分别为 GSM8K 0.45337/0.45792、MATH500 0.188/0.194，没有超过 standard。`beta=0.05` 下 sampled-token penalty 信号很小，且 `ref_penalty_standard` 出现 max grad norm 1144。
- pass@4/diversity 子集实验显示，长训退化不是简单 diversity collapse：GSM8K-256 上 base mean/pass@4 为 0.4766/0.6523，standard200 降到 0.3779/0.5430，但 unique answer rate 从 0.6211 升到 0.6826，pairwise Jaccard distance 从 0.5280 升到 0.6260；MATH200 也有同样趋势。topk200 相对 standard200 略提高 pass@4，同时降低多样性，说明 reliability routing 可能在抑制错误扩散。

当前判断：

可靠性 gate 是有价值的稳定性机制，但不能宣称它单独提升 accuracy。已观察到 gating 从“去噪”转为“欠学习”的区间：strict gate 明显降低梯度，却伤害 MATH100；full split 上 top-k routing 也没有稳定超过 standard；200-step 下它只能减轻长训漂移；robust regrade 排除了“只是 boxed grader 太严”的主要疑虑。reference anchoring 和 explicit reference penalty 进一步说明，sampled-token 近似 trust region 只能部分影响轨迹，且会有 run-to-run variability。pass@4/diversity 结果又说明漂移不是简单 mode collapse，而是更多低质量路径的错误扩散。当前最值得写成论文主线的版本不是“RC-OPD 提升准确率”，而是“用 reliability signals 诊断并控制 OPD drift/variance”，包括 early stopping、sequence-level trust region 或 adaptive step-size。

## Idea 2: ExOPD 需要局部可信度约束，而不是全局 lambda > 1

核心问题：G-OPD/ExOPD 证明 lambda > 1 的 reward extrapolation 有可能超过教师，但 λ 外推也会放大教师隐式奖励中的噪声。

方法：比较三种 λ 策略：

- full ExOPD: 所有 token `lambda=1.25`
- routed ExOPD: 可靠 token `lambda≈1.25`，不可靠 token 降权
- conservative OPD: `lambda=1`

为什么有研究价值：

- ExOPD 的理论告诉我们“外推可能有收益”，但 OPD 失败模式文献告诉我们 teacher signal 局部不可靠。
- 这两者之间存在直接张力：外推强度越大，对 teacher reliability 的要求越高。

可行性：

- 实现代价低，只需在已有 OPD loss 中暴露 λ。
- 本次 pilot 已经观察到风险：full adaptive_exopd 在 GSM100 上为 0.390，低于 base 0.430；routed_adaptive 为 0.400，略好于 full adaptive_exopd 但仍低于 base。MATH100 上 full adaptive_exopd 回到 base 的 0.220，而 routed_adaptive 为 0.260，略高于 standard/topk 的 0.250。
- 较小外推强度 `lambda=1.10` 更安全：GSM100 为 0.440，高于 `lambda=1.25` routed_adaptive 的 0.400；MATH100 为 0.250，低于 `lambda=1.25` routed_adaptive 的 0.260 但高于 full adaptive_exopd 的 0.220。
- 训练上，full adaptive_exopd 的平均梯度范数最高 146.1，max grad norm 426；routed_adaptive 把 max grad norm 降到 310，但没有阻止性能退化。
- full split 上 `lambda=1.10` 的任务依赖性更明显：GSM8K 为 0.41926，是四个候选中最高但只比 base 高 0.23 个百分点；MATH500 为 0.160，与 base 相同且低于 standard 的 0.168。

当前判断：

这是一个很好的负结果方向：不是简单复述 ExOPD，而是研究“外推什么时候不该做”。当前证据说明 `lambda>1` 的风险主要来自外推目标本身；reliability gate 能缓和梯度尖峰，并可能在部分分布上保留收益，但当前形式不足以让外推在不同任务上稳定安全。合理的研究问题应从“lambda > 1 是否有效”改为“lambda 应如何随 task/trajectory/local reliability 自适应”。

## Idea 3: Teacher 选择需要 local compatibility + capability gap 双轴

核心问题：更大或更高 benchmark 的 teacher 未必在学生实际访问的 prefix 上提供不同且有用的局部目标；但反过来，最贴近学生的 teacher 也未必最有用，因为它可能缺少足够的可迁移能力差距。

方法：对候选教师做无训练诊断：

- student rollout 固定；
- 多个 teacher 在相同 prefixes 上打分；
- 比较 top-k overlap、teacher margin、entropy、implicit reward 方差、teacher-teacher KL；
- 同时估计 teacher capability gap，例如 teacher 本身在任务上的 pass@1、teacher continuation accuracy、或 verifier score；
- 用 local compatibility 和 capability gap 的二维指标预测 OPD 是否会有收益。

为什么有研究价值：

- `Rethinking OPD` 的 distributionally indistinguishable teacher 现象说明 benchmark 分数不足以预测 OPD 成败。
- 多教师 OPD/MOPD 会遇到 teacher routing 问题，局部可区分性比全局榜单更接近 OPD 的训练信号。
- 本次 1.7B teacher 反例说明 local compatibility 也不是充分条件：更像学生的 teacher 可能更稳定，但不一定更有能力。

可行性：

- 诊断部分很便宜：只需要 rollout + teacher scoring。
- 本次已下载官方 `Qwen/Qwen3-1.7B`，固定 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，并完成 0.6B/1.7B/4B 三模型对比。
- 96-example 三分布诊断显示，1.7B 比 4B 更局部兼容：train/GSM8K/MATH500 上 1.7B top-k overlap with student 为 0.647/0.653/0.682，4B 为 0.614/0.618/0.657；1.7B abs advantage 为 0.534/0.527/0.307，4B 为 0.622/0.620/0.362。
- 但 1.7B-teacher 50-step OPD 给出反例：GSM100 上 standard/topk 为 0.410/0.390，低于 4B-teacher 的 0.460/0.440，也低于 base 0.430；MATH100 上为 0.240/0.260，与 4B-teacher 0.250/0.250 接近。
- full robust regrade 保留了这个任务依赖性：1.7B-teacher OPD 在 GSM8K 上 standard/topk 为 0.440/0.449，低于 base 0.462 和 4B-teacher OPD 0.462/0.462；MATH500 上为 0.194/0.204，其中 topk 略高于 base 0.202 和 4B-topk 0.194。
- teacher direct generation 也不是充分解释：robust regrade 下 1.7B/4B teacher 自身 GSM8K 为 0.440/0.458，MATH500 为 0.168/0.154。direct-generation 分数与 OPD gain 相关但不完全一致。
- 因此成果不应只是单一 teacher selection metric，而应是二轴或 Pareto scoring：compatibility 太低不可学，capability/task gap 太低不值得学。

当前判断：

高可行，但必须避免“overlap 越高越好”的错误结论。当前最有价值的形式是 measurement + controlled OPD gain correlation：用官方 1.7B/4B/更多同家族教师画出 compatibility-capability 二维图，解释为什么 1.7B 更兼容却不一定更好，为什么 4B 在 GSM8K 上提供更多可迁移增量，而 1.7B top-k 在 MATH500 上反而略占优。

## Idea 4: Trajectory-then-token Reliability Filtering

核心问题：token-level gate 只看局部，不知道整条轨迹是否偏离正确推理方向；trajectory filtering 只看整条，又可能丢掉局部有用 token。

方法：

1. trajectory hard filter：用 teacher average log-prob / verifier / answer correctness 丢掉底部 rollout；
2. token soft weight：在保留轨迹内用 reliability gate 调整 token 更新强度；
3. 对错误轨迹只在 first-divergence suffix 或 high-confidence correction span 上蒸馏。

为什么有研究价值：

- FiRe-OPD 支持 “trajectory hard filter + token soft reweight” 的组合。
- TOPD 支持 reasoning failure 是轨迹级现象，不只是孤立 token 错误。
- RC-OPD 的 pilot 显示单 token gate 不足以保证最终 accuracy，因此需要 trajectory-level context。

可行性：

- 可在当前 clean-room trainer 上扩展：先给每条 rollout 计算 trajectory score，再传给 token loss。
- GSM8K/MATH 可以用 answer correctness 当 cheap trajectory signal。
- 不需要新模型，只需多一次 scoring/eval。
- 已验证的简单版本是负结果：正确轨迹权重 0.5、错误轨迹权重 1.5 的 `outcome_reweight` 在 200-step full GSM8K/MATH500 上为 0.334/0.126，未超过 topk_router 的 0.339/0.132，也明显低于 base。整条错误轨迹放大并不够。
- early-stopping curve 表明，trajectory signal 的第一用途可能不是决定“学哪些错误轨迹”，而是决定“何时停止/降学习率/收紧 KL trust region”。

当前判断：

这是 RC-OPD 最自然的下一版，但必须避免“整轨迹粗放 scaling”。研究问题应改为：如何定位 first-divergence span、只蒸馏 verifier-supported correction span，或把 answer correctness 变成 early stopping/trust-region 信号，而不是把错误 trajectory 全部放大。当前证据更支持后两者。

## Idea 4.5: Explicit Trust-Region OPD for Drift Control

核心问题：sampled-token OPD 在某些训练轨迹中 50 到 100 step 之间出现明显退化，但另一些 100-step run 又能在 GSM8K 上维持甚至略超 base；这说明问题不只是局部 token 信号噪声，也包括训练轨迹选择和策略整体 trust region。简单 reference-gap anchoring 降低了梯度尖峰，但无法恢复 base 性能；explicit sampled-token reference penalty 在当前 beta 下也没有超过 standard。

方法：

- 显式约束 `KL(pi_current || pi_base/reference)`，而不是只在 sampled token 上扣 reference gap；
- 用 reliability / entropy / outcome 指标动态调整 KL coefficient 或停止训练；
- 当 teacher-student overlap 下降、entropy 激增或 robust validation 下滑时收紧 trust region。

为什么有研究价值：

- OPD 文献强调 on-policy 能减少 distribution shift，但我们的曲线显示 on-policy sampled-token training 仍会产生后训练漂移。
- G-OPD 把 OPD 解释为 KL-constrained RL 特例；因此 trust-region 控制不是额外技巧，而是 OPD/RL 等价关系中的自然自由度。
- f-OPD / StableOPD / Decoupling KL 等工作都从不同角度指出 stale rollout、entropy collapse、KL mixing 是稳定性核心问题。

可行性：

- 当前 clean-room trainer 已能加载官方 Qwen3-0.6B reference，并已验证 `anchored_standard` / `anchored_topk_router` / `ref_penalty_standard` / `ref_penalty_topk_router` 的 checkpoint、resume 和 full eval。
- 100-step anchor 结果给了明确的调参方向：`anchored_topk_router` 将 max grad norm 降到 138，并把 GSM8K robust primary 提到 0.41926，但 MATH500 仍低于 base。这说明信号存在，但 sampled-token correction 不够。
- 100-step explicit penalty 结果是负/混合证据：standard 本身 GSM8K robust primary 为 0.46702，而 ref-penalty 变体没有超过它；`ref_penalty_standard` 还出现 max grad norm 1144。这说明“加一个 sampled-token L2 penalty”不是足够的 trust region。
- 下一版只需在 loss 中增加 batch/sequence-level KL estimate 或 periodic held-out validation trigger，不需要新模型或新数据。

当前判断：

这是最直接承接当前负结果的方向。它比继续调 top-k threshold 更有价值，因为退化表现为训练轨迹和 trust-region 问题；也比直接上 latent OPD 低风险。目标应定义为“降低 OPD 训练轨迹方差并延长有效训练窗口”，而不是立即追求大幅 accuracy gain。

## Idea 5: Diversity-Preserving OPSD

核心问题：OPSD/self-teacher 会把 privileged context、hint、sampled demonstration 的风格一起蒸馏进去，导致 style drift 和输出多样性收缩。

方法：

- correct-hint vs wrong-hint contrast 去掉共同 style drift；
- evidence mask 只在 privileged evidence 支撑的 token/span 上蒸馏；
- 加入 entropy / semantic diversity monitor，对 pass@k 曲线而不只 pass@1 做 early stopping。

为什么有研究价值：

- EDGE-OPD、RLCSD、OPSDL 都指向同一问题：privileged/self signal 并非全 token 可信。
- OPD 主线的 teacher reliability 问题，在 OPSD 中变成 self-teacher signal contamination。

可行性：

- 可用公开 GSM8K/MATH 构造 correct/wrong hint，不依赖闭源 teacher。
- 评估可以用 pass@k、distinct reasoning pattern、answer diversity。
- 当前 eval 脚本已有 `n` rollouts 参数，可扩展为 diversity 指标。
- 本次 OPD pass@4/diversity 子集实验已验证评估管线可用，但也提醒“多样性变化”方向不一定是收缩：standard200 比 base 有更高 unique answer rate 和 token Jaccard distance，却有更低 pass@4。因此 OPSD 的多样性研究不能只追求更高/更低 diversity，而要区分 useful diversity 与 error-spreading diversity。

当前判断：

可行但比 RC-OPD 多一个数据构造环节。适合作为第二篇或主论文的 extension。指标应同时报告 pass@k、answer diversity、semantic/path diversity 和 verifier-filtered diversity，否则容易把错误扩散误读成健康探索。

## Idea 6: Selective Latent/Token OPD

核心问题：token-level OPD 受离散 token 支撑集限制，长链推理中教师 token-level advantage 会退化。

方法：

- 高确定性、非关键步骤走 latent hidden-state alignment；
- 低确定性、计算关键或格式关键步骤保留 token KL；
- gate 可由 entropy、margin、step verifier 或 intrinsic dimensionality 触发。

为什么有研究价值：

- Coconut/CODI/SLT 显示 latent reasoning 与 selective compression 有潜力。
- OPD 机制论文指出 token 空间瓶颈和长序列崩塌，这正是 latent route 的动机。

可行性：

- 工程成本最高：需要抽 hidden states、决定对齐层、处理 teacher/student hidden dimension 或架构一致性。
- 同家族 Qwen3 student/teacher 可能让 hidden-state 对齐更可行。
- 初步可做 frozen diagnostic：比较 correct vs wrong rollouts 的 hidden-state CKA/MSE，不必一开始训练 latent policy。

当前判断：

高风险高收益，不适合作为当前第一篇 pilot paper 的主线；适合后续深入项目。

## Idea 7: Token-Stride / Sparse-Token OPD

核心问题：标准 OPD 对 completion 中每个 token 都计算 sampled-token supervision，但 OPD 并未规定必须监督所有 token。是否可以每隔 2 或 3 个 token 监督一次，在保持效果的同时降低计算？

方法：

- `stride=1`: dense token OPD，原始目标；
- `stride=2`: 只监督一半 completion tokens；
- `stride=3`: 只监督约三分之一 completion tokens；
- 训练仍每 step 更新一次学生权重，只改变参与 loss 的 token 子集。

为什么有研究价值：

- 论文5/论文7 都指出 token-level OPD 的有效信号高度集中，长尾 token 可能贡献噪声或高方差；
- 如果 sparse-token supervision 能保持效果，就说明 OPD 不需要全 token dense supervision；
- 如果进一步实现 sparse teacher scoring，可能降低 teacher-side 计算成本。

可行性：

- 已在 clean-room trainer 中实现 `--token-supervision-stride` 和 `--token-supervision-offset`，默认 `stride=1` 不改变旧实验；
- `stride=2` resume gate 通过，checkpoint 完整；
- 官方 Qwen3-0.6B student + 官方 Qwen3-4B teacher 的 50-step 对照已经完成。

当前证据：

- 监督 token rate 按预期下降：stride1/2/3 分别为 1.000/0.500/0.336；
- 但当前标准 forward 实现不省 wall-clock：sec/step 为 6.942/6.976/6.965；
- 梯度范数升高：avg grad norm 为 115.06/151.69/201.23，max grad norm 为 244/398/560；
- full robust regrade 不降反升：GSM8K 为 0.4617/0.4670/0.4678，MATH500 为 0.198/0.200/0.204；
- pilot GSM100/MATH100 也没有显示明显崩溃：stride3 与 dense 在 GSM100 持平，在 MATH100 略高。

当前判断：

“少监督 token 仍能保持质量”这个弱版本初步成立；“直接省计算”这个版本在当前实现中不成立。因为我们只是 mask loss，teacher/student 仍完整 forward。真正有价值的后续方向是 sparse teacher scoring 或 chunked/prefix-cache teacher scoring：只在选中 token 位置计算 teacher log-prob/top-k，并验证是否能在 wall-clock 或显存上带来实际收益。该方向值得作为 RC-OPD 主线的效率 extension，但需要多 seed 验证，因为当前 accuracy 差异幅度很小。
