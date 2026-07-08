# Block Reverse-KL OPD Idea

## 用户修正后的 idea

当前 OPD 通常按 token-level sampled log-prob 做监督。这里的新想法不是“每隔几个 token 取一个 token”，而是把连续 2 或 3 个 token 作为一个 block，让 teacher 对这个 block 的联合概率打分，再用 block-level reverse-KL / implicit reward 更新 student。

换句话说，监督单位从：

```text
token: y_t
```

变成：

```text
block: y_{t:t+k-1}
```

其中 block 的 teacher log-prob 是 autoregressive 条件概率的和：

```text
log pi_T(y_{t:t+k-1} | prefix)
= sum_{i=t}^{t+k-1} log pi_T(y_i | prefix, y_{t:i-1})
```

student / old policy / reference policy 同理。

## 与已完成 token-stride 实验的区别

已完成的 `token_stride_opd_experiment.md` 做的是 sparse-token supervision：

- `stride=2`: 只监督位置 0,2,4,...
- `stride=3`: 只监督位置 0,3,6,...

它仍然是 single-token sampled objective，只是丢掉了一部分 token。它不构造连续 token block，也没有把多个 token 的 log-prob 合成一个 block-level score。

因此 token-stride 实验不能回答 block reverse-KL OPD 是否有效，只能说明“少监督 token 是否还能保持效果”。

## 当前 trainer 是否符合该 idea

不符合。

当前 `clean_opd_train.py` 的关键行为：

1. `token_logprobs()` 返回每个 completion token 的 sampled log-prob；
2. `token_logprobs_topk()` 也返回每个 token 的 teacher/student sampled log-prob 和 top-k；
3. `raw_advantage = teacher_logp - old_logp` 是逐 token 计算；
4. loss 是逐 token 的 `-(weight_t * advantage_t * current_logp_t)` 求和；
5. 新增的 `token_supervision_stride` 只是 mask 掉一部分 token。

因此它仍是 token-wise OPD，不是 block-wise OPD。

## Block OPD 的可实现版本

最直接的 sampled-block 版本：

```text
teacher_block_logp = sum teacher_logp[t:t+k]
old_block_logp     = sum old_logp[t:t+k]
current_block_logp = sum current_logp[t:t+k]

block_advantage = teacher_block_logp - old_block_logp
loss = - block_weight * stopgrad(block_advantage) * current_block_logp
```

这相当于让 teacher 对 student 实际生成的连续 k-token block 做联合打分。

需要明确两种不同目标：

1. sampled-block OPD：只对 student 采样到的 block 打分，工程可行；
2. exact block reverse KL：对所有长度 k 的 token 组合求 KL，空间是 `|V|^k`，对 LLM 基本不可行，只能近似。

## 预期价值

- block-level 信号能把局部 token 的影响和短程 future coupling 绑在一起；
- 可能减少单 token 信号的噪声；
- 能研究 OPD 的监督粒度：token、block、trajectory 三者之间是否存在更优中间粒度；
- 若结合 sparse block scoring，可能进一步减少 teacher scoring 位置。

## 风险

- 如果只是 full forward 后把 log-probs sum 成 block，wall-clock 不会明显下降；
- block advantage 方差可能随 block size 增大；
- block 边界选择会影响结果：non-overlap blocks、sliding blocks、random blocks 可能不同；
- 需要重新设计 reliability gate：block reliability 应该由 block 内 overlap、teacher entropy、margin 或 min/mean confidence 聚合。

## 建议下一步实验

实现 `--opd-granularity token|block` 和 `--block-size 2/3`：

- baseline: token OPD (`block-size=1`);
- block2 non-overlap;
- block3 non-overlap;
- 可选：sliding block2/3。

先跑 50-step clean-room pilot，比较：

- wall-clock；
- gradient norm；
- GSM100/MATH100；
- full GSM8K/MATH500 robust regrade；
- block advantage variance。

如果 block2/3 在质量上不掉，下一步再做真正的 sparse/block teacher scoring 优化。
