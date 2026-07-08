# Token-Stride OPD Experiment

## 问题

用户提出：当前 OPD 是 token-level objective，但不一定要每个 token 都监督。是否可以只看每 2 或 3 个 token 中的一个 token，再更新学生权重，从而降低计算量或保持效果？

## 实现

在 `clean_opd_train.py` 中加入：

- `--token-supervision-stride`
- `--token-supervision-offset`

`stride=1` 为原始 dense token OPD。`stride=2/3` 表示只在 completion token 位置 `i % stride == offset` 上计算 OPD loss。默认值保持 `stride=1`，不改变已有实验。

注意：当前实现仍用标准 Transformer forward 计算完整序列 logits，然后在 loss 上 mask token。因此它验证的是“稀疏 token supervision 是否有效”，不是一个真正节省 teacher forward FLOPs 的优化实现。真正要省算力，需要进一步改 teacher scoring / logits projection / prefix batching。

## 工程验证

使用官方模型：

- Student/reference: `Qwen/Qwen3-0.6B`
- Teacher: `Qwen/Qwen3-4B`

不使用用户自训 `Qwen3-MoE`。先做 `stride=2` resume gate：step1 保存，resume 到 step2，checkpoint 完整性检查通过。

## 训练设置

三组 50-step 对照：

- `dense_stride1`: `stride=1`
- `sparse_stride2`: `stride=2`
- `sparse_stride3`: `stride=3`

共同设置：`max_new_tokens=128`, `prompts_per_step=2`, `lr=5e-7`, `top_k=20`, official 0.6B student + official 4B teacher。

## 训练统计

| variant | supervised token rate | sec/step | avg grad norm | max grad norm |
|---|---:|---:|---:|---:|
| dense_stride1 | 1.000 | 6.942 | 115.06 | 244 |
| sparse_stride2 | 0.500 | 6.976 | 151.69 | 398 |
| sparse_stride3 | 0.336 | 6.965 | 201.23 | 560 |

结论：监督 token 数确实按预期降低，但 wall-clock 没有下降；梯度范数反而更高。

## Pilot Evaluation

Robust regrade:

| variant | GSM100 | MATH100 |
|---|---:|---:|
| dense_stride1 | 0.46 | 0.25 |
| sparse_stride2 | 0.43 | 0.27 |
| sparse_stride3 | 0.46 | 0.26 |

## Full Evaluation

Robust regrade:

| variant | GSM8K | MATH500 |
|---|---:|---:|
| dense_stride1 | 0.4617 | 0.198 |
| sparse_stride2 | 0.4670 | 0.200 |
| sparse_stride3 | 0.4678 | 0.204 |

Simple grader:

| variant | GSM8K | MATH500 |
|---|---:|---:|
| dense_stride1 | 0.4185 | 0.162 |
| sparse_stride2 | 0.4162 | 0.164 |
| sparse_stride3 | 0.4238 | 0.166 |

## 当前判断

该 idea 的“质量保持”部分初步成立：50-step 下 stride2/3 没有伤害 full GSM8K/MATH500，stride3 在 robust regrade 上略高于 dense baseline。

该 idea 的“省计算”部分在当前实现中不成立：虽然 loss 只监督 1/2 或 1/3 token，但标准 HF forward 仍处理完整序列，因此 sec/step 基本不变。

最有价值的下一步不是继续只改 loss mask，而是实现真正的 sparse teacher scoring：只计算选中位置的 teacher log-prob / top-k，或用 chunk/prefix cache 复用来降低 teacher-side cost。同时需要多 seed 验证，因为当前差异幅度较小。
