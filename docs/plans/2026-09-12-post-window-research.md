# Window 结果之后的研究路线草案

状态：2026-09-12；不是新训练的启动协议。当前只恢复 Random3/Sliding3
Step50/100/200 的完整评测。新增四卡 ml2 机时预算已询问用户，尚未确认。
没有下载新模型、生成新数据、启动新 seed 或扩大既有队列。

## 1. 现有证据决定能说什么

- 本轮 Random3 历史 grader 的 macro Avg@8 从 Step50 的 17.84% 降至
  Step200 的 9.18%；Pass@8 从 39.95% 降至 34.31%。Sliding3 尚未测完。
- [历史跨师生结果](../results/2026-07-13-block3-cross-pair-validation.md)：
  Qwen3 配对中 Block3 有单 seed 正结果；DeepSeek/JustRL 配对未复现。
  旧结果只作背景，不能拼入本轮 Sliding3-vs-Random3 的配对统计。
- “数学恒等式成立”“运行无 NaN”“最终准确率提高”是三种不同的结论。
  论文目标应是可复现的贡献或边界解释，不是保证每一组都涨点。

当前主分析仍是 Step200 的 Sliding3 - Random3；Step50/100 不替换主终点。
即使前者显著优于后者，也只说明优于 Random3，不代表优于 Token/Fixed3。

## 2. 优先核查：mean 是否真正消除了更新尺度差异

记 A_t = teacher_logp_t - old_student_logp_t，ell_t 为当前学生 log-prob。
对有效长度为 m_W 的窗口，当前实现的未裁剪局部目标是：

```text
L_phase = -(1/N) sum_W [m_W * mean(A_W) * exp(sum_W(current - old))]
```

其中 N 是当前 loss 归约单元的有效 token 数。令 current=old，则：

```text
C_t = -N * dL_phase/dell_t = sum_{j in W(t)} A_j
```

即每个 token 的实际输入导数系数仍是窗口 SUM。不能把诊断记录中的
shared mean 当作真实导数。实现已经同时保留二者，不需要改写历史实验。
这里讨论的是 log-prob 输入导数，不是完整模型参数梯度。

Sliding3 对三个偏移的完整 loss 平均。在远离 mask/EOS/首尾边界的位置：

```text
C_t = (A_{t-2} + 2*A_{t-1} + 3*A_t + 2*A_{t+1} + A_{t+2}) / 3
```

该核的系数和为 3，不是 1。局部 A 恒定为 a 时，Token 的系数为 a，
当前 Fixed3/Sliding3 内部位置的系数为 3a。

2026-09-12 本地 CPU/FP64 autograd 检查（12 个有效 token，A 全为 1）：
Token 全为 1，Fixed3 全为 3，Sliding3 为
`[2, 8/3, 3, 3, 3, 3, 3, 3, 3, 3, 8/3, 2]`。
这是受控导数检查，不是新的模型训练或准确率实验。

它不证明“3 倍梯度就是 collapse 原因”：Adam 的尺度响应、梯度裁剪、
参数共享及训练中的 ratio 都会改变实际更新。下一阶段必须测量实际
update norm 与 clipping，不能只把 LR 除以 3 就宣布控制完了混杂。

## 3. 候选 A：单位增益的邻域信用分配

假设：短距离 advantage 含有可共享信息，但无条件 joint ratio/长度权重
混入了尺度变化。先只改 credit，保留 Token OPD 的 per-token PPO ratio。

```text
M_t = sum_j w_{t,j} A_j
w >= 0, sum_j w_{t,j} = 1
A_t_new = M_t
```

首个候选用固定三角核 `[1,2,3,2,1]/9`，在每个连续有效 response 内计算，
边缘只对有效权重重新归一化，不跨 EOS、padding 或 mask 间断。
它不是“同时预测五个 token”，仍复用同一次 teacher forward 的 log-prob。
所有 A 和权重 stop-gradient。只改变 sampled-OPD surrogate 的信用系数，
不能直接称为原 sequence reverse-KL 的无偏梯度。

可行性来自一个有边界的去噪假设：若邻域真实信号相同、噪声独立且方差
同为 sigma^2，归一化三角核的噪声方差为 `19/81 * sigma^2`。
真实推理的相邻 token 往往相关，正确表达应为 `w^T Sigma w`，并加上
平滑引入的偏差平方。因此降噪与涨点均需实测，不能套用独立噪声结论。

最小新增训练对照建议为 contemporaneous Token vs 候选 A，从相同 Base
独立开始，固定 prompt schedule、训练步数和全部评测设置。先一个配对
seed 作探索，不把历史 Token 当作这次的完整因果对照。若预算无法覆盖
两组训练及三轮完整评测，先不启动，不能中途缩小 eval 以节省成本。

## 4. 候选 B：有界邻域修正，而不是强制所有 token 共享信号

仅在 A 显示邻域信息可能有用、但边界/异号污染仍明显时考虑：

```text
A_t_new = A_t + lambda * clip(M_t - A_t, -c*abs(A_t), c*abs(A_t))
0 <= lambda <= 1, 0 <= c <= 1
```

候选默认值可在新协议中固定为 lambda=0.5、c=1，而非在当前测试集上扫参。
它限制邻域修正幅度；A_t 非零时不翻转其符号，A_t=0 时不更新。
但“保持原符号”不等于“保持正确方向”：原 teacher signal 也可能错误，
近零信号被抑制可能损害探索。应与无约束归一化平滑直接比较，而不是
只与更弱的 Random3 比较。该候选未实现、未验证，也未确认独立新颖性。

为什么仍值得检验：它不需要新模型、teacher generation 或额外 verifier，
只增加 O(T) 的张量运算，能在同一 codebase 中隔离邻域信用的收益与污染。
不能将这点宣传为已证实的训练加速。

## 5. 新颖性边界与外部对照

- [BPDG](https://arxiv.org/html/2606.24084v1) 用 old-current policy drift 做
  block 门控；我们拟检验 teacher-old advantage 在邻域间如何分配。
  它用了两个 PPO epochs、AMC23 40 题，与当前一个 epoch、83 题不同，
  论文表中的数字不能直接拿来比较。
- [SG-OPD](https://arxiv.org/html/2606.09304v1) 已用 verifier/teacher 的符号
  一致性决定外推或回退；仅提出“按符号 gating”不足以构成新贡献。
  候选 B 针对邻域共享前后的符号、不增加 verifier，但这只是区别，
  尚不是新颖性或有效性的证明。
- [FiRe-OPD](https://arxiv.org/abs/2606.02684) 已结合轨迹过滤与 token
  重加权；[Prune-OPD](https://arxiv.org/abs/2605.07804) 已用局部兼容性
  衰退控制奖励和 rollout。不能把通用 reweighting/overlap gating 当创新。
- 外部基线优先从现有 [Revisiting OPD](https://arxiv.org/abs/2603.25562)
  codebase 复现作者的 local-support-matching 配置，冻结版本与 top-K 等
  参数。适配到同一实验合同后标为“统一配置复现”，不是抄论文原分数。
- 若研究主张涉及信用分配优于短期回报，必须加入 truncated reward-to-go/
  GAE 对照。若涉及联合 ratio 稳定性，再加入 BPDG；不要无关地堆基线。

当前更值得形成的贡献是：解释局部信用共享何时有效、何时被尺度或异号
污染抵消，并给出通过必要对照的简单方法，而不是单独包装 block size=3。

## 6. 泛化路线：保留反例，再扩大容量和训练分布

| 维度 | 建议设置 | 回答的问题 |
|---|---|---|
| 已有正结果 | Qwen3-1.7B-Base <- Qwen3-4B-Base-GRPO | 多 seed 下还能否复现 |
| 已有反例 | DeepSeek-R1-Distill-Qwen-1.5B <- JustRL-DeepSeek-1.5B | 失败是否稳定、机制能否预测 |
| 可选第三对 | 官方 Qwen/Qwen3-4B-Base <- 同一公共 Qwen3-4B-Base-GRPO | 更换学生容量、等规模能力迁移 |
| 第二训练数据 | AI-MO/NuminaMath-CoT 的去污染 prompt-only 子集 | 是否只适用于当前 DAPO prompt 分布 |

[Qwen3-4B-Base 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B-Base)
可核验学生身份。教师仍是研究者公开的 GRPO checkpoint，不改称 Qwen
官方后训练发布；新增 pair 的兼容性与显存可行性需单独门禁。

[NuminaMath-CoT 数据卡](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
提供 problem/source 等字段。未来只用题目，不拿其 CoT 作 teacher target。
先锁定 revision，排除与全部 eval/dev 题的精确及近重复、审计高风险来源，
再固定子集和 manifest。最终题目数需处理后记录，不能提前声称已去污染。
对 DAPO 的新实验也应用同一去污染标准；旧 raw-pool 结果原样保留并标注。

不要一次做方法 x 师生 x 数据 x seed 全因子：优先原 pair 的多 seed，
然后已知反例和第二数据；第三对仅在预算充足时加入。新训练 seed 不
因低分替换，失败 run 不删。三次训练 seed 与八次 eval 采样必须分开统计。

## 7. 论文证据包与阶段门槛

1. 当前队列：两组各三轮完整结果、历史 grader 输入/输出哈希、固定
   Step200 配对 CI、训练/评测成本、热图和格式错误；未完成项明确留空。
2. 机制筛选：同一冻结 batch 上比较 Token/Fixed3/Sliding3/归一化版本的
   输入导数，测量实际参数 update 与裁剪。已有输入空间诊断不能替代
   尚未做的完整参数梯度探针。
3. 新方法探索：固定公式和 dev 选择规则；Token、归一化对照先成对比较。
   只有优于当期 Token，才进入 Fixed3/有界修正/相关外部方法的必要消融。
4. 确认性阶段：预先登记 practical-effect 门槛、seed 数和预算；至少规划
   三个配对 training seeds，报告每 seed 和训练方差。三 seed 仍可能统计
   功效不足，不能保证显著性。测试题 bootstrap 不能代替训练重复。
5. 泛化与效率：第二 pair、第二训练数据、长度/截断/熵/语义多样性、
   GPU 时间和 teacher 调用量。只涨 Avg@8 却明显损害 Pass@8 的结果应
   作为权衡报告，不模糊成全面提升。

完整评测沿用既定四任务 n=8、Step50/100/200，不能以“pilot”之名只测
有利题目。数学公式证明只能支持局部性质；当前公开测试集已经多次
被查看，后续方法选择应使用独立 dev，最终增设未参与选择的锁定测试集。

条件决策：Sliding3 若胜 Random3，先查是否只是缓解差基线的退化；若
两者都退化，优先归一化/信用消融，不继续盲目扩展窗口。Block3 多 seed
或跨 pair 仍不稳时，将结果转为条件有效性与失败机制研究，不隐藏反例。
当前还没有足以宣称“新方法有效”或“实验已足够成文”的完整证据。
