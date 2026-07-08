# 2026-07-08 Block Advantage Experiment

## Goal

Validate whether block-level reverse-KL OPD can reduce supervision units while
preserving quality, and whether weighting/normalizing token advantages fixes
the gradient amplification seen in naive block aggregation.

## Run

- Remote run root: `/mnt/data/cpfs/Yaleon/opd_cleanroom_20260707/runs/block_advantage_20260708_144223`
- Student: official `Qwen/Qwen3-0.6B`
- Teacher: official `Qwen/Qwen3-4B`
- Training: 50 steps per variant
- Eval: full GSM8K 1319 examples and full MATH500 500 examples
- Final scoring: post-hoc regrade

## Results

| Variant | Advantage aggregation | Grad norm mean | Grad norm max | GSM8K | MATH500 |
|---|---|---:|---:|---:|---:|
| naive block-3 | `A1 + A2 + A3` | 466.74 | 1136.00 | 0.4655 | 0.192 |
| mean block-3 | `(A1 + A2 + A3) / 3` | 165.22 | 504.00 | 0.4693 | 0.198 |
| mixed block-3, lambda=0.5 | `0.5 Ai + 0.5 mean(A1,A2,A3)` | 83.29 | 175.00 | 0.4708 | 0.188 |

## Conclusion

The idea is supported in the variance-controlled form. Mean block-3 reduced
gradient norm by roughly 65% versus naive block-3 and slightly improved both
GSM8K and MATH500.

The mixed objective gave the lowest gradient norm but lost MATH500 accuracy in
this setting, so it needs a lambda sweep before it can be treated as a better
default.

The current research framing should be:

```text
Block-level reverse-KL OPD needs variance-controlled advantage aggregation.
```

not:

```text
Larger block size is automatically better.
```
