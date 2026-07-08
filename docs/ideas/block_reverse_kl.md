# Block Reverse-KL OPD Idea

Standard sampled-token OPD supervises each sampled token with its own
teacher-vs-old-student advantage. The block idea asks whether several
consecutive sampled tokens can share one block-level reverse-KL signal.

For a sampled sequence:

```text
y = (y_1, ..., y_T)
```

Token-level OPD uses:

```text
A_t = log pi_teacher(y_t | x, y_<t) - log pi_old(y_t | x, y_<t)
L_token = - sum_t stopgrad(A_t) * log pi_theta(y_t | x, y_<t)
```

A naive 3-token block uses:

```text
A_B = A_1 + A_2 + A_3
L_block = - stopgrad(A_B) * (logp_1 + logp_2 + logp_3)
```

The current evidence suggests that naive block aggregation amplifies gradients.
The more promising version controls the block advantage scale:

```text
A_B = (A_1 + A_2 + A_3) / 3
```

or mixes token-local and block-level signals:

```text
A_mix,i = (1 - lambda) A_i + lambda mean(A_1, A_2, A_3)
```

The research question is therefore not simply whether larger blocks help, but
how to aggregate block-level advantages without losing credit assignment or
inflating gradient variance.
