# Literature and Mathematical Audit: Historical Block3 Mean OPD

Audit date: 2026-09-21. This worker independently audited literature and mathematics. Only this file and ../references.bib are authored here. No controller/manuscript edits, SSH, GPU execution, training, model downloads, or commits. Other methods/history workers retain their ownership.

Source checkout: 6fcd9e3e373861da39055a7540d669b829b80a2f. The core loss, actor, sampled-block launcher, and diagnostics have no diff from frozen runtime 0f9161f02f08287fb07f0375ad0a6bda81133ff0. This establishes source equivalence for those files, not execution provenance for every historical run. The dispatching-parallel-agents ownership boundary was followed; no callable agent-dispatch tool was available in this worker.

## 1. Conclusions for the Writers

1. Historical **Block3 mean is a joint-action PPO variant**, not just an advantage smoother. It averages teacher-minus-old-student scores, sums current/old log-probabilities before exponentiating, and reduces over fractionally weighted block units.
2. For a full three-token block with equal advantages, at current=old on the active loss branch, each log-probability derivative is approximately three times sampled-token's derivative. Generally there are cross-token score terms, changed clipping, and normalization effects, not a uniform learning-rate multiplier.
3. The current short tail is retained with weight valid_count/3. The old cleanroom drops it and uses weighted log-probabilities without a PPO ratio. Its "mixed" mode also has a different meaning.
4. **Teacher calls are not reduced.** Grouping occurs after token scoring; every rollout position is still teacher-scored. Fewer loss units do not demonstrate fewer forward calls, lower FLOPs, or a wall-clock speedup.
5. Pathwise log-probability factorization is exact. Calling the practical clipped, masked, top-p-sampled, frozen-advantage training objective "exact KL" is not justified. A restricted idealized first-order interpretation is available below.
6. Conditional scalar MSE reduction needs explicit signal/covariance assumptions. It does **not** imply universal gradient-variance reduction, stability, or downstream gains.
7. "First blockwise OPD", "first selective OPD", and "unbiased sequence-KL gradient" are unsupported. A defensible contribution requires controlled evidence about this specific joint-ratio/credit-assignment/reduction design, including failures and scale confounds. This bounded audit is not a priority certificate.

The final bibliography is restricted to 15 entries. Requested keys: gkd, minillm, revisiting, rethinking, bpdg, ppo, dagger, dapo, qwen3, llama3. Additional keys: gspo, r2opd, math, prm800k, llama32. Other inspected near neighbors remain linked in this internal audit, not added to the main bibliography.

Preprint entries use first-posting years and versioned URLs. GKD and MiniLLM notes separately record ICLR 2024. MiniLLM's verified v6 title is **MiniLLM: On-Policy Distillation of Large Language Models**; its earlier title was not independently checked here. Llama 3's author field intentionally abbreviates the current list after three verified names. Llama 3.2 model-card evidence is separate from the Llama 3/3.1-era technical report.

## 2. Implementation Evidence Map

Paths are relative to the repository root; anchors refer to the audited checkout.

| Evidence | Read location | Consequence |
|---|---|---|
| OPD advantages | external/revisiting_opd/verl/trainer/ppo/core_algos.py:889 | Teacher-minus-old-student sampled score; optional paths are not all enabled. |
| Actual OPD arguments | external/revisiting_opd/verl/trainer/ppo/ray_trainer_multitask.py:421 and :1880 | Reads nested algorithm.opd.gamma, default 0, not generic algorithm.gamma=1; reward weight defaults 0. |
| Fixed blocks | external/revisiting_opd/verl/trainer/ppo/core_algos.py:461 | Fixed tensor positions, sum log-probabilities, mean advantages, fractional tail mask. |
| PPO clipping | external/revisiting_opd/verl/trainer/ppo/core_algos.py:523 | Exponentiate the block log-ratio before clipping; negative-advantage dual clip c=3. |
| Loss reduction | external/revisiting_opd/verl/trainer/ppo/core_algos.py:395 | "token-mean" acts on the supplied block arrays after aggregation. |
| Numerical denominator | external/revisiting_opd/verl/utils/torch_functional.py:169 | masked_mean divides by mask.sum()+1e-8. |
| Effective mask/settings | external/revisiting_opd/verl/workers/actor/dp_actor.py:114, :1353, :1380, :1406 | Response/loss mask, optional entropy mask and special-token exclusion; final kl_mask enters policy loss. |
| Accumulation | external/revisiting_opd/verl/workers/actor/dp_actor.py:1507 | Fixed microbatches: divide each independently reduced loss by accumulation count. |
| Historical defaults | scripts/run_revisiting_sampled_block_opd_math.sh:31, :149, :175 | top-p=.9, temperature=1, unclamped OPD signal, special-token masking false, added KL loss false, entropy coefficient 0, one PPO epoch. |
| Config defaults | external/revisiting_opd/verl/trainer/config/ppo_trainer.yaml:49 | Dynamic batching false; gradient-norm clip 1; PPO low/high .2; dual clip 3; token-mean. |
| Old cleanroom | scripts/clean_opd_train.py:181, :555, :590 | Drops incomplete blocks; frozen weighted advantages times current log-probability; no PPO ratio. |
| Credit diagnostics | opd_ext/diagnostics.py:86 | Sign flips and averaging discrepancy, not full gradient discrepancy or causal harm. |

The aggregation docstring's statement about keeping the mean signal scale close to token OPD concerns the **advantage scalar**, not equality of gradients. Its phrase "sampled block reverse-KL" is not a proof of an exact divergence objective.

### 2.1 Targets and Masks

For history \(h_t=(x,y_{<t})\), p_theta is the student scoring distribution, p_0 the stored old-student scoring distribution, q the fixed teacher, and mu the actual sampler. Define

\[
 l_t^\theta=\log p_\theta(y_t\mid h_t),\quad
 l_t^0=\log p_0(y_t\mid h_t),\quad l_t^q=\log q(y_t\mid h_t),
 \qquad A_t=l_t^q-l_t^0. \tag{M1}
\]

The audited sampled baseline has no reward addition, temporal return (OPD gamma=0), advantage whitening, or teacher-log-ratio clipping. Old/teacher arrays are prepared outside the differentiable actor update and treated as frozen targets. The advantage helper itself does not explicitly detach, so do not attribute a detach operation to that helper.

Let m_t be the **final actor mask**, binary for these derivations. It starts from response attention positions (or multi-turn loss_mask), may incorporate an incoming entropy mask, and may exclude special tokens when enabled. This launcher sets opd_mask_special_tokens=False; do not claim all special tokens are removed. EOS/padding inclusion follows the upstream mask. For binary masks the earlier response-mask multiplication in advantage construction is idempotent.

Fixed blocks are groups of k tensor positions from offset zero, **without compressing mask holes**. For k>1 invalid entries are zeroed with torch.where before multiplication, and the right edge is zero-padded to a multiple of k. Define

\[
 n_b=\sum_{t\in b}m_t,\quad w_b=n_b/k,\quad
 \bar A_b={\sum_{t\in b}m_tA_t\over\max(n_b,1)},\quad
 \Delta_b=\sum_{t\in b}m_t(l_t^\theta-l_t^0),\quad R_b=e^{\Delta_b}. \tag{M2}
\]

Empty blocks have zero advantage/weight. Incomplete blocks are retained. R_b is a product, **not** the geometric mean exp(Delta_b/n_b). Its likelihood-ratio interpretation needs the conditions in Section 4.

The "sum" option uses sum(m_t A_t). Current "mixed" uses \((1-\lambda)\sum m_tA_t+\lambda\bar A_b\). Neither is the historical mean treatment. For k<=1 the helper returns raw token inputs. Random/sliding windows and full/top-k KL are separate paths; compatibility checks reject k>1 with GSPO or full/top-k KL.

### 2.2 Exact Loss, Clip, and Reduction

Let \(d=1-\epsilon_{\rm low}=.8\), \(u=1+\epsilon_{\rm high}=1.2\), and c=3. The minimized scalar loss is

\[
 C(r,a)=\max\{-ar,-a\,\operatorname{clip}(r,d,u)\},\qquad
 \ell(r,a)=
 \begin{cases}\min\{-ca,C(r,a)\},&a<0,\\ C(r,a),&a\ge0.\end{cases} \tag{M3}
\]

Thus "clip .2" alone omits the negative-advantage dual clip. Away from kinks, a>0 has active gradient for r<u; a<0 for d<r<c under these settings. Clipping is a surrogate-loss operation, not a hard constraint on the next policy's KL or ratios. Original PPO defines the ordinary clipped surrogate, not this additional dual-clip term. [PPO, §3, PDF p.3, Eq.7](https://arxiv.org/pdf/1707.06347v2).

Let N be the number of valid tokens within one reduction group and \(\eta=10^{-8}\). The **production** reductions are

\[
 L_{\rm tok}={\sum_tm_t\ell(r_t,A_t)\over N+\eta},
 \quad r_t=e^{l_t^\theta-l_t^0},\qquad
 L_{\rm blk}={\sum_bw_b\ell(R_b,\bar A_b)\over\sum_bw_b+\eta}
 ={ \sum_bn_b\ell(R_b,\bar A_b)\over N+k\eta}. \tag{M4}
\]

The last equality is real-arithmetic algebra; dtype rounding still applies. Empty finite-input groups reduce to zero. The theoretical calculations below explicitly set eta=0 and N>0. A two-token tail for k=3 has weight 2/3, not 1.

**Microbatching matters.** Historical actor microbatch size is 1, mini-batch size 32, PPO epochs 1. Fixed batching accumulates equally weighted, independently normalized response losses, with distributed averaging. Ignoring eta,

\[
 L_{\rm update}={1\over M}\sum_{i=1}^M{1\over N_i}
       \sum_b n_{ib}\ell(R_{ib},\bar A_{ib}), \tag{M5}
\]

for equally weighted response microbatches. This is not the batch-global-token objective \(\sum_{ib}n_{ib}\ell/\sum_iN_i\) when lengths differ. The configuration string "token-mean" does not establish DAPO-style global token weighting. [DAPO, §3.3, PDF p.6, Eq.12](https://arxiv.org/pdf/2503.14476v2).

Other agg_loss modes also operate on post-block arrays: sequence-mean-token-sum is \(M^{-1}\sum_{ib}w_{ib}\ell_{ib}\); sequence-mean-token-mean is \(M^{-1}\sum_i(\sum_bw_{ib}\ell_{ib})/\sum_bw_{ib}\), without the masked_mean epsilon; sequence-mean-token-sum-norm is \(\sum_{ib}w_{ib}\ell_{ib}/B_{\rm padded}\), where B_padded is the block-axis size. These are not the audited default and change scaling/empty-row behavior.

### 2.3 First-Order Difference from Sampled-Token

For valid t in block b, on the active unclipped branch and with frozen targets/masks,

\[
 {\partial L_{\rm blk}\over\partial l_t^\theta}
 =-{n_b\bar A_bR_b\over N+k\eta},\qquad
 {\partial L_{\rm tok}\over\partial l_t^\theta}
 =-{A_tr_t\over N+\eta}. \tag{M6}
\]

At current=old both ratios equal 1. Broadcasting the block mean into **token-ratio, token-normalized** PPO instead gives \(-\bar A_b/(N+\eta)\). The implemented coefficient is larger by \(n_b(N+\eta)/(N+k\eta)\), approximately n_b.

Let \(s_t=\nabla_\theta l_t^\theta|_{\theta_0}\). With eta=0 and sums over valid positions,

\[
 g_{\rm blk}=-{1\over N}\sum_b
   \left(\sum_{j\in b}A_j\right)\left(\sum_{t\in b}s_t\right),
 \quad g_{\rm tok}=-{1\over N}\sum_tA_ts_t,
 \quad
 g_{\rm blk}-g_{\rm tok}
 =-{1\over N}\sum_b\sum_{\substack{j,t\in b\\j\ne t}}A_js_t. \tag{M7}
\]

Scalar losses coincide at ratio 1 for eta=0, but derivatives do not. If advantages are constant within every full block, k=3 gives \(g_{\rm blk}=3g_{\rm tok}\). Generally this equality fails: tails have n_b<k, signs vary, and shared network Jacobians interact. Adam, global gradient clipping, optimizer history, and later PPO clipping also prevent concluding that parameter updates are always three times larger.

CPU checks of the real loss helper with the production masked_mean, float64:

| Synthetic input | Token result | Block3 mean result |
|---|---|---|
| A=(1,1,1), all r=1 | Loss approximately -1; gradients (-1/3,-1/3,-1/3) | Loss approximately -1; gradients (-1,-1,-1) |
| A=(1,1,1), each r=1.1 | Loss approximately -1.1; each gradient approximately -.3667 | R=1.331; loss approximately -1.2; gradients zero |
| Five tokens, A=1,r=1 | Each gradient approximately -.2 | First three approximately -.6; tail two approximately -.4 |
| A=(1,-1,1), r=1 | Gradients approximately (-1/3,+1/3,-1/3) | Gradients approximately (-1/3,-1/3,-1/3) |
| Four positions, mask=(1,0,1,1), A=1,r=1 | Selected gradients approximately -1/3 | Approximately (-2/3,0,-2/3,-1/3) |
| A=(-1,-1,-1), each r=2 | Negative loss remains active at r=2 | R=8 exceeds c=3; loss approximately +3; gradients zero |

These are derivative checks, not measured training ratios. Production eta gives -0.9999999900000002 instead of -1 in the first block row. Existing policy-loss tests stub masked_mean without eta; this audit separately loaded its actual AST to check the denominator, mask holes, all-zero masks, dual clipping, and centered finite differences (error below 1e-9 away from kinks).

## 3. Old Cleanroom Is a Different Objective

Let T be completion length, \(U=\lfloor T/k\rfloor\), \(T'=Uk\). Let alpha_t denote its reliability/extrapolation multiplier (code name lambdas), distinct from the mixed-mode interpolation parameter, and v_t its token weight. Cleanroom's raw \(A_t^c\) is teacher-minus-old-student log-probability **clamped to [-20,20]** before unit construction.

Mean-mode blocks contain only the first T' positions:

\[
 \widetilde A_b=k^{-1}\sum_{t\in b}\alpha_tA_t^c,\quad
 \widetilde v_b=k^{-1}\sum_{t\in b}v_t,\quad
 L_{\rm clean}=-{1\over\max(1,U)}\sum_{b=1}^U
   \operatorname{sg}(\widetilde v_b)\operatorname{sg}(\widetilde A_b)
   \sum_{t\in b}l_t^\theta . \tag{M8}
\]

Here sg denotes frozen supervision. Optional reference penalties are additional terms. If U=0 the builder emits no units and training can skip the example.

| Property | Current historical block PPO | Old cleanroom mean |
|---|---|---|
| Mean signal | Frozen sampled advantages; no launcher target clamp | Multiplier-scaled, [-20,20]-clamped advantages |
| Current log-probabilities | Sum, then exp(current-old) | Sum, used linearly |
| PPO ratio/clipping | Joint ratio, ordinary and dual clipping | Neither |
| Denominator | Sum(n_b/k)+1e-8 per microbatch | Number of full block units, at least 1 |
| Tail | Retained, fractional weight | Dropped |
| mixed | Block sum/mean interpolation; one block unit | Per-token local/mean interpolation; token units and denominator T' |
| Nondefault token stride | Outside this fixed-block audit | Disallowed for k>1 |

For v=alpha=1, full blocks, matching targets, eta=0, and current=old, the two mean methods share a local gradient. Scalar losses and subsequent optimization differ. Cleanroom mixed with v=1 is
\(-T'^{-1}\sum_{t\le T'}[(1-\lambda)\alpha_tA_t^c+\lambda\widetilde A_{b(t)}]l_t^\theta\),
not the current block sum/mean interpolation.

## 4. KL and Bias: Exact Statements and Their Conditions

Sections 4-5 contain independent audit derivations, not claims of novel theorems. They distinguish factorization, expectation, and optimization gradient.

### 4.1 Pathwise Identity versus Objective

At a fixed block-start prefix h, for an unmasked block of specified length,

\[
 \log{P_\theta(Y_b|h)\over Q(Y_b|h)}
 =\sum_{t\in b}\log{p_\theta(y_t|h_t)\over q(y_t|h_t)}. \tag{M9}
\]

This is exact factorization. Under \(Y_b\sim P_\theta(\cdot|h)\), compatible support and an existing expectation, its expectation is conditional block reverse KL. Individual samples can be negative. Averaging rescales a fixed-length log-ratio; clipping or selecting actions changes its interpretation.

Full-vocabulary conditional KL at a **fixed sampled prefix** can be evaluated exactly. Teacher-top-K-renormalized KL is exact for restricted distributions, not automatically full-vocabulary KL. Neither implies exact full-sequence gradients if derivatives through the prefix distribution are omitted. GKD explicitly stops sampling gradients (§3.1, PDF p.4, Eq.4); Revisiting distinguishes token and sequence gradients (§2.1, Appendix D). [GKD](https://arxiv.org/pdf/2306.13649v3), [Revisiting](https://arxiv.org/pdf/2603.25562v2).

### 4.2 Fixed-Horizon Gradient and Boundary Bias

Assume fixed T, no masking/truncation, fixed q positive on student support, compatible token IDs, true p_theta sampling, integrability permitting differentiation under expectation, and no PPO/target clipping. Let \(c_t=\log p_\theta-\log q\), \(s_t=\nabla\log p_\theta\), \(C=\sum_tc_t\), \(S=\sum_ts_t\). Then

\[
 \nabla D_{\rm KL}(P_\theta\Vert Q)
 =\mathbb E[CS]+\mathbb E[S]=\mathbb E[CS]
 =\mathbb E\left[\sum_{t=1}^Ts_t\sum_{j=t}^Tc_j\right]. \tag{M10}
\]

For j<t, c_j is measurable from h_t and \(\mathbb E[s_t|h_t]=0\). Past terms vanish in expectation, not necessarily in variance.

At theta=theta_0 with frozen A_t=-c_t, define unnormalized estimators

\[
 H_{\rm tok}=\sum_tc_ts_t,\quad
 H_{\rm blk}=\sum_b\left(\sum_{j\in b}c_j\right)\left(\sum_{t\in b}s_t\right),
 \quad H_{\rm seq}=CS.
\]
\[
 \mathbb E[H_{\rm blk}]
 =\mathbb E\left[\sum_ts_t\sum_{j=t}^{e(t)}c_j\right],\quad
 \mathbb E[H_{\rm seq}-H_{\rm blk}]
 =\mathbb E\left[\sum_ts_t\sum_{j>e(t)}c_j\right], \tag{M11}
\]

where e(t) is the end of t's block. The idealized implemented gradient is H_blk/T. Token supervision drops all future terms; block supervision restores within-block future credit but drops terms across boundaries. The missing vector need not shrink monotonically with k: terms can cancel and different partitions need not nest.

A narrower unbiased interpretation holds **at theta_0**. Freeze each block-start prefix distribution \(d_b^0\), and define
\(F(\theta)=T^{-1}\sum_b\mathbb E_{h_b\sim d_b^0}D_{\rm KL}(P_\theta(Y_b|h_b)\Vert Q(Y_b|h_b))\).
Under the same assumptions, \(\mathbb E[H_{\rm blk}/T]=\nabla F(\theta_0)\). Later block-start distributions are not differentiated. With k=T this is full fixed-horizon sequence KL/T; away from theta_0, frozen old targets and clipping give a surrogate, not generally the current KL gradient.

Eight-path Bernoulli enumeration checked this ideal identity. For t=0,1,2, the student logit was \(\theta+.45\sum_{j<t}y_j-.2t\); the teacher logit was \(-.7+.9\sum_{j<t}y_j-.2t\), with theta=.3. Sequence KL/3 had derivative 0.1635621418328663. Expected helper gradients at eta=0 were 0.17997144833328366 (k=1), 0.1730201028391579 (k=2), and 0.16356214183286621 (k=3). This is not evidence of monotonicity in general. For production eta at fixed T, multiply each value by T/(T+k eta).

### 4.3 Conditions Not Automatically Met in This Runtime

- **Top-p sampling:** at .9 and temperature 1, ideally \(\mu_t(v|h)=p_0(v|h)\mathbf1\{v\in S(h)\}/Z(h)\), while the stored denominator scores p_0. Generally \(\mathbb E_\mu[\nabla\log p_0]\ne0\). Truncation removes support, so ordinary reweighting of retained samples cannot recover all full-p mass. Kernel/scoring discrepancies can add mismatch. Both arms share this qualification.
- **Prefix occupancy:** even a correct conditional block ratio does not correct preceding actions' distribution. A product over masked positions is generally not the likelihood ratio of a block marginal when intermediate actions are omitted.
- **EOS and normalization:** random N and division by N can depend on future actions. Zero-mean past terms multiplied by future-dependent 1/N need not remain zero. M10-M11 require fixed T; stopped/length-normalized objectives need a separate EOS/censoring derivation.
- **Frozen/clipped targets:** old advantages, ordinary/dual clip, optional log-ratio clamps, reward mixing, nonzero OPD gamma, or selected masks require their own bias claims.
- **Token identity:** teacher/student tokenizer and special-token compatibility must be checked for actual artifacts; shared family names alone do not prove it.
- **One epoch:** does not guarantee ratio=1 throughout; rollout/scoring differences and updates can induce drift. Post-update diagnostics cannot retroactively constrain the optimizer. Logged ppo_kl is a masked old-current sampled log-ratio diagnostic, not teacher-student KL.

Rethinking's tokenwise KL/value discussion (Eq.2-3) does not contradict Revisiting's gradient-bias discussion: an unbiased value estimator need not yield the full sequence gradient when differentiated as a frozen-sample surrogate. MiniLLM already explicitly incorporates future rewards (§2.2-2.3, PDF pp.3-5), so within-block future credit is not the first recognition of temporal OPD credit assignment. [Rethinking](https://arxiv.org/pdf/2604.13016v2), [MiniLLM](https://arxiv.org/pdf/2306.08543v6).

## 5. Conditional MSE, Variance, and Leakage

### 5.1 A Scalar Smoothing Result, Not a Universal Guarantee

Condition on information F that fixes the block, its underlying signal vector, and covariance model. Write \(A_t=\alpha_t+\varepsilon_t\), with \(\mathbb E[\varepsilon_t|F]=0\), finite conditional covariance \(\Sigma\), and n valid positions. For a token-specific target alpha_t,

\[
 \operatorname{MSE}(\bar A_b,\alpha_t\mid F)
 =(\bar\alpha_b-\alpha_t)^2+{1\over n^2}{\bf1}^{\mathsf T}\Sigma{\bf1},
 \quad
 \operatorname{MSE}(A_t,\alpha_t\mid F)=\Sigma_{tt}. \tag{M12}
\]

Therefore smoothing improves that conditional MSE **if and only if**
\((\bar\alpha_b-\alpha_t)^2 < \Sigma_{tt}-n^{-2}{\bf1}^{\mathsf T}\Sigma{\bf1}\).
The right-hand side may be nonpositive. With constant local signal and independent equal-variance noise, it becomes sigma^2/n. With equicorrelation rho it becomes
\(\sigma^2[1+(n-1)\rho]/n\); perfectly correlated noise yields no reduction. For position-dependent signals, averaging introduces bias even when the noise averages out.

Under \(|\operatorname{Cov}(\varepsilon_i,\varepsilon_j|F)|\le C_{|i-j|}\), with variance at most sigma^2 and summable nonnegative C_h,

\[
 \operatorname{Var}(\bar\varepsilon_b|F)
 \le {n\sigma^2+2\sum_{h=1}^{n-1}(n-h)C_h\over n^2}
 =O(1/n). \tag{M13}
\]

This is not a fitted model of the actual log-ratios. Conditioning on adaptively selected spans or actions can invalidate assumed mean-zero errors/covariance bounds. R2-OPD already states a related geometrically decaying-covariance result for span means (Proposition 2, PDF p.4, Eq.9-10); basic mean-variance reduction is not an unclaimed novelty. [R2-OPD](https://arxiv.org/pdf/2608.19408v1).

Crucially, the implemented active coefficient is \(n\bar A_b/(N+k\eta)\), not \(\bar A_b/(N+\eta)\). With eta=0, fixed N, and independent equal-variance noise, its noise variance is \(n\sigma^2/N^2\), versus \(\sigma^2/N^2\) for one token coefficient. Comparing n*barA to the token target alpha_t also changes the bias to n*baralpha-alpha_t. Thus M12 cannot be transferred to this coefficient, much less to the shared parameter gradient, without additional analysis.

### 5.2 Gradient Variance and Ratio Sensitivity

Gradient variance means variability at a **fixed policy and sampling protocol**, e.g. trace(Cov(H)). It is not the mean gradient norm across training steps. A gradient norm incorporates signal, cancellations, changing parameters, clipping, and scaling; even a smaller norm does not establish lower estimator variance.

For M11, if \(|c_t|\le C_0\), \(\|s_t\|\le S_0\), fixed T, and block lengths at most k, then

\[
 \|H_{\rm blk}\|\le C_0S_0\sum_bn_b^2\le C_0S_0kT,\quad
 \operatorname{tr}\operatorname{Cov}(H_{\rm blk})
 \le C_0^2S_0^2k^2T^2. \tag{M14}
\]

For H_blk/T the bound is \(C_0^2S_0^2k^2\). This uses no independence, but the boundedness assumptions do not follow from this runtime: its teacher log-ratio clamp is disabled, and score norms are not automatically bounded. These worst-case upper bounds, including token versus sequence bounds, are **not an ordering of actual variances**. Adding zero-expectation past terms can increase or decrease variance depending on covariance. Revisiting's Appendix D (PDF pp.15-16) similarly supplies bounded worst-case bounds; its Appendix E.1-2 synthetic measurement is a separate empirical result. [Revisiting](https://arxiv.org/pdf/2603.25562v2).

For old-current log shifts delta_t, \(\log R_b=\sum_t\delta_t\), so
\(\operatorname{Var}(\log R_b)=\sum_{i,j}\operatorname{Cov}(\delta_i,\delta_j)\).
There is no 1/n averaging in this ratio. Correlated shifts can amplify block drift or cancel; exponentiation and shared clipping change active-gradient patterns. The r_t=1.1 example in Section 2 shows all tokens individually active while the block is saturated. PPO bounds its surrogate branches, not every likelihood ratio or optimizer displacement. Neither clipping nor an average advantage alone proves stable updates.

### 5.3 What the Existing Leakage Diagnostics Measure

Let \(E=\{t:m_t=1,\ |A_t|>10^{-4},\ |\bar A_{b(t)}|>10^{-4}\}\), and let F_flip contain eligible positions with opposite signs. The code reports

\[
 f_{\rm sign}={|F_{\rm flip}|\over\max(|E|,1)},\quad
 f_{\rm weighted}={\sum_{t\in F_{\rm flip}}|A_t|
                         \over\max(\sum_{t\in E}|A_t|,10^{-4})},
\]
\[
 D_{\rm mean}={\sum_tm_t|\bar A_{b(t)}-A_t|\over\max(N,1)},\quad
 D_{\rm norm}={\sum_tm_t|\bar A_{b(t)}-A_t|
                         \over\max(\sum_tm_t|A_t|,10^{-4})}. \tag{M15}
\]

These describe redistribution of sampled advantages. They are not error rates against correct reasoning, causal harm measures, or complete loss-gradient discrepancies. In particular, equal advantages give zero leakage but approximately n-fold coefficients. At ratio 1 and eta=0,

\[
 n_b\bar A_b-A_t=(\bar A_b-A_t)+(n_b-1)\bar A_b,\qquad
 \|g_{\rm blk}-g_{\rm tok}\|
 \le {1\over N}\sum_tm_t|n_b\bar A_b-A_t|\,\|s_t\|. \tag{M16}
\]

The logged discrepancy sees only the first component. Actual eta changes the comparison to
\(n_b\bar A_b/(N+k\eta)-A_t/(N+\eta)\). Some cross-token terms encode legitimate future credit under M11; calling every such term "bias leakage" would also be misleading.

## 6. Primary-Source Claim Map and Novelty Boundary

This is a bounded search centered on the requested papers and close optimization-granularity neighbors, completed through 2026-09-21. Paper versions are pinned below. No claims from local review HTML were accepted without checking the primary paper. No comprehensive priority search or independent reproduction of external experiments is claimed.

| Primary source | Supported claim and precise locator | Boundary for this project |
|---|---|---|
| [GKD, Agarwal et al., 2306.13649v3](https://arxiv.org/abs/2306.13649v3) | Student-generated traces, alternative divergences, and stopped sampling gradients: §2-3.1, PDF pp.3-4, Eq.2-4, Algorithm 1; RL combination §3.2, p.5, Eq.5. | Neither on-policy distillation nor teacher feedback on student prefixes is new. GKD's distributional loss is not identical to this sampled-token PPO baseline. |
| [MiniLLM, Gu et al., 2306.08543v6](https://arxiv.org/abs/2306.08543v6) | Reverse sequence KL, immediate/future reward decomposition, teacher-mixed sampling and practical importance approximations: §2.1-2.3, PDF pp.3-5, Eq.1-7, Algorithm 1. | Do not characterize all earlier OPD as independent-token-only. Practical approximations are not a proof of exact KL optimization here. |
| [Revisiting, Fu et al., 2603.25562v2](https://arxiv.org/abs/2603.25562v2) | Sampled-token failure modes and truncated local teacher support matching: §2.1-3.1, PDF pp.2-6, Eq.1-8. Student and teacher are both renormalized on teacher top-K support. | This is a restricted-support distributional baseline, not merely averaging sampled advantages. Prefix unreliability and token mismatch are established concerns. |
| [Rethinking, Li et al., 2604.13016v2](https://arxiv.org/abs/2604.13016v2) | Thinking-pattern/teacher-capability observations: §3.1 p.5; high-probability overlap: §4 pp.9-11; cold start/prompt alignment: §5 pp.11-14; long-horizon discussion: §6 pp.14-16. | 97-99% mass is setup-specific; overlap cardinality is not mass. The anisotropy discussion in §6.2 is explicitly unverified, not a causal theorem. Suffix-to-prefix degradation cannot be assumed for our runs. |
| [BPDG, Zheng and Jiang, 2606.24084v1](https://arxiv.org/abs/2606.24084v1) | Found through reports/blockwise_policy_drift_gating_review.html, then verified from the paper. §3 pp.3-4, Eq.1 uses old-current student drift: detached g_b=exp(-tau*abs(mean_b(delta_t))), normalized to unit mean over valid positions. | Gate reweights original token losses; teacher targets/support and token denominator stay unchanged. It neither broadcasts teacher advantages nor uses our joint PPO ratio. At current=old its gates are uniform, unlike our changed gradient. |
| [GSPO, Zheng et al., 2507.18071v2](https://arxiv.org/abs/2507.18071v2) | Length-normalized sequence ratio exp(mean log-ratio): §3-4.2, PDF pp.3-4, Eq.5-10; token extension §4.3, Eq.13-17. | Existing sequence-level ratio/clipping work; our exp(sum log-ratio) is not GSPO's geometric ratio. Calling Block3 "GSPO on shorter sequences" would omit a material distinction. |
| [R2-OPD, Yang et al., 2608.19408v1](https://arxiv.org/abs/2608.19408v1) | Span construction, independent continuation-based progress ranking, selective filtering: §3-4, PDF pp.3-5; remaining-token reduction Eq.13 p.5. | Already studies span-level OPD reliability and covariance-conditional mean variance. Its verifier-based filtering does not establish correctness of arbitrary fixed-width averaging. |
| [PPO, Schulman et al., 1707.06347v2](https://arxiv.org/abs/1707.06347v2) | Sampled policy-gradient surrogate §2.1 p.2; clipping §3 p.3; minibatch reuse Algorithm 1 p.5. | Cite for ordinary PPO; do not describe clipping as an exact trust-region constraint or credit this paper with the extra dual-clip branch. |
| [DAgger, Ross, Gordon, Bagnell](https://proceedings.mlr.press/v15/ross11a.html) | Publisher abstract motivates learning on policy-induced observation distributions. | Conceptual antecedent only. Its guarantees were not read/proved applicable to LLM distillation. Use AISTATS 2011 publication year; preprint first posted 2010. |
| [DAPO, Yu et al., 2503.14476v2](https://arxiv.org/abs/2503.14476v2) | RL loss/clip/dynamic sampling and token-global reduction: §2.2-3.4, PDF pp.3-6. | Source for RL/dataset lineage, not evidence that this OPD variant is universally stable or uses the same effective denominator. Dataset name does not identify unique row counts. |

BPDG is particularly close but not equivalent. Its §4 uses two PPO epochs and fixed 64-token/newline spans; §7 (p.6) limits evidence to one pair/seed and does not claim unbiased off-policy correction. Its AMC23 evaluation has 40 problems (p.5), whereas the local contract has 83; scores should not be directly compared as the same benchmark artifact. Signed-mean drift can cancel, and mean-normalized gates can exceed 1. These are different control semantics, not an established superiority of either design. [BPDG, §§3-4,7](https://arxiv.org/pdf/2606.24084v1).

### 6.1 Additional Close Neighbors: Internal Links Only

These remain outside the 15-entry bibliography to avoid turning the main paper into a large survey. Metadata was verified against arXiv, not copied from the local reports.

- **SOD: Step-wise On-policy Distillation for Small Language Model Agents**, Qiyong Zhong, Mao Zheng, Mingyang Song, Xin Lin, Jie Sun, Houcheng Jiang, Xiang Wang, Junfeng Fang (2026), [2605.07725v3](https://arxiv.org/abs/2605.07725v3). Read PDF pp.3-5, §3-4.3, Eq.5-10: tool-delimited response steps, stepwise divergence weights multiplying token losses, combined with GRPO. No joint block PPO product in those equations. SNR propositions were not independently verified; do not reuse them as established guarantees.
- **Filter, Then Reweight: Rethinking Optimization Granularity in On-Policy Distillation**, Yuying Li, Leqi Zheng, Yongzi Yu, Wenrui Zhou, Xuchang Zhong, Xing Hu, Jing Jin, Hangjie Yuan, Tao Feng (2026), [2606.02684v2](https://arxiv.org/abs/2606.02684v2). Read pp.3-5, §3.1-3.2, Eq.2-9: trajectory filtering followed by soft within-response token weighting and token PPO. Unit-mean weights do not by themselves fix gradient norm. Official spelling is **Hangjie**, not the "Huangjie" found in some secondary metadata.
- **TIP: Token Importance in On-Policy Distillation**, Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, Alborz Geramifard (2026), [2604.14084v4](https://arxiv.org/abs/2604.14084v4). Read pp.3-4, §3-4, Eq.1-3 and Table 1: token selection based on student entropy and teacher disagreement. This precludes claiming that nonuniform informativeness of OPD positions is new. Experimental/theoretical claims beyond these sections were not audited.
- **SG-OPD: Sign-Gated On-Policy Distillation via Sign-Consistency Gating and Phased Teacher Sampling**, Haoran Xu, Hongyu Wang, Yifei Gao, Jiaze Li, Xiaofeng Zhang, Xiaosong Yuan (2026), [2606.09304v1](https://arxiv.org/abs/2606.09304v1). Read pp.3-4, §3-4.2, Eq.1-8: verifier/teacher sign routing and phased verified teacher sampling. Its sign test is not our token-versus-block-mean diagnostic; its signal sign convention must be reconciled with loss direction before comparison.

**Novelty verdict.** Generic block aggregation, OPD reliability weighting, multi-step credit, and scalar averaging bounds are occupied claim areas. The read methods do not establish identity with this particular combination of mean advantages, unnormalized joint block ratio, fractional-tail weighting, and per-response reduction; that is a scoped distinction, not proof of first publication. The 3x scaling effect alone is an implementation observation, not sufficient algorithmic novelty.

Useful controls, not claimed to have been run here: token-ratio PPO with broadcast block means; block-ratio PPO with an explicitly matched coefficient scale; a geometric-mean ratio control; length/normalizer-matched comparisons; independent repeated-rollout gradient estimates at a fixed checkpoint; multiple training seeds/pairs. Dividing learning rate by k is not a clean substitute because Adam, clipping, tails, and ratio geometry remain different.

## 7. Model, Data, and Evaluation Sources

### 7.1 Model Identity Must Be Version-Specific

- **qwen3:** [Qwen3 Technical Report, 2505.09388v1](https://arxiv.org/abs/2505.09388v1), An Yang et al. (2025). Read PDF pp.11-15; §4.3/Table 9 p.11 covers thinking-mode formatting, §4.5 p.12 strong-to-weak distillation. This is official family/mode evidence, not proof that a separately GRPO-tuned teacher is the official Qwen3-4B post-trained model.
- The local public teacher path lllyx/Qwen3-4B-Base-GRPO currently redirects to [Thinking-Space/Qwen3-4B-Base-GRPO](https://huggingface.co/Thinking-Space/Qwen3-4B-Base-GRPO). Its Model Description/Training Details identify Qwen3-4B-Base plus GRPO, associated with Rethinking. Repository revision read via API: 1f3b2966edfb75f2f98a00617588c1f748088422. The card is the uploader's primary artifact source, **not** an official Qwen team release or certification of local weights. Keep a direct card footnote rather than another bibliography entry.
- **llama3:** [The Llama 3 Herd of Models, 2407.21783](https://arxiv.org/abs/2407.21783), Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. (2024), current metadata accessed. Only abstract/metadata were read. This Llama 3/3.1-era report is family background, not a same-version reference for Llama 3.2 1B/3B checkpoints.
- **llama32:** [Meta's Llama-3.2-1B-Instruct model card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), Model Information, explicitly covers the Llama 3.2 1B/3B release and September 25, 2024 release date. Revision: 9213176726f574b556790deb65791e0c5aa438b6. Distinguish Base from Instruct in protocol text. The local Llama diagnosis identifies historical Instruct checkpoints; neither the family paper nor this card establishes local checksum identity. No weights were downloaded.

### 7.2 MATH and MATH-500 Are Different Evaluation Artifacts

**math:** Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt, **Measuring Mathematical Problem Solving With the MATH Dataset** (2021), [2103.03874v2](https://arxiv.org/abs/2103.03874v2). Read PDF pp.3-5, especially §3.1 pp.4-5: 12,500 competition problems split 7,500 train/5,000 test, seven subject categories, boxed-answer extraction and normalization-based automatic evaluation. This does not specify a 500-item test split or guarantee correctness of arbitrary symbolic graders. [Author repository README](https://github.com/hendrycks/math) also explicitly identifies loaders and evaluation code.

**prm800k:** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe, **Let's Verify Step by Step** (2023), [2305.20050v1](https://arxiv.org/abs/2305.20050v1). Paper abstract/metadata only; no full-paper claim. The official [PRM800K README, MATH Splits and Answer Grading](https://github.com/openai/prm800k#math-splits) was read: its nonstandard split leaves 500 MATH test problems for evaluation, with the other 4,500 added to training. It describes normalization plus SymPy-based grading and acknowledges possible grading errors. The [HuggingFaceH4 MATH-500 card](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/raw/main/README.md) explicitly links this source. The card/viewer identifies 500 test rows, but local file identity still requires manifest/ID/hash comparison by the protocol worker.

For AMC/AIME, the [MAA competition organizer page](https://maa.org/student-programs/amc/) verifies competition identity, not a particular ML evaluation subset, deduplication policy, year/forms mapping, or grader. No fabricated "AMC23 paper" or unverifiable benchmark citation was added. The 40-versus-83 discrepancy above must be resolved by artifact names/counts, not hidden under one label.

For n generated answers per problem, report separately the per-sample accuracy \((Pn)^{-1}\sum_{i,j}z_{ij}\) and the empirical problem solve rate \(P^{-1}\sum_i\mathbf1\{\sum_jz_{ij}>0\}\). With n=8 the latter is an observed pass@8/solve@8 statistic, not majority voting. These definitions do not certify the local implementation. Independent-sample assumptions, repeated seeds, parse failures, and empty/truncated outputs require explicit protocol handling. Benchmark identity and decoding/grading parity matter before any external score comparison.

## 8. Historical Evidence Read: Limits, Not a New Results Report

The history worker remains responsible for final historical tables. This audit read the following local artifacts to prevent unsupported method claims:

| Artifact | Read scope | Allowed inference |
|---|---|---|
| docs/results/2026-07-08-block-advantage.md | Full text | Short cleanroom mean-versus-sum pilot; reported gradient norms do not measure fixed-policy estimator variance. |
| docs/results/2026-07-13-block3-cross-pair-validation.md | Full text | The positive Qwen historical signal did not establish cross-pair universality; other pair results constrain claims. |
| docs/results/2026-09-19-llama-block3-quality-diagnosis.md | Full text | Template/think-mode mismatch, repeated training samples, actual block gradient/reduction, and degradation chronology are confounds, not a single proven cause. |
| docs/plans/2026-07-11-ml2-block3-replication-design.md | Initial fixed-contract section | Historical planned protocol, not proof of completed execution. |
| reports/blockwise_policy_drift_gating_review.html | Source/ID and relevant text search | Discovery of 2606.24084 only; the PDF, not the review, supports literature claims. |
| reports/block_opd_experiment_report.html; reports/block3_ml2_replication.html | Relevant searched excerpts, not complete HTML review | Context pointers; no unverified external bibliography imported. |
| results/2026-07-11-block10-collapse-dual-host.json | Full JSON | Local snapshots report collapse on two hardware contexts; not a live-host check or proof of a unique mechanism. |

No new training outcome, live completion state, or pooled benchmark score is asserted. No claim of systematic suffix-to-prefix collapse follows from these snapshots. Training-step averages of gradient norms are not estimator-variance measurements. A paired problem bootstrap does not capture training-seed variation. Exact-hash contamination checks do not exclude semantic overlap, and a repeated-row data pool is not a count of unique questions.

## 9. Manuscript-Ready Wording

These are independent wording suggestions, not edits to either manuscript. Equations M1-M6 specify the exact implementation including eta; paper exposition may omit eta only after identifying it as a numerical denominator guard.

**English, methods.** Both treatments teacher-score every token of their own student-generated trajectories; grouping a given trajectory into blocks does not reduce teacher calls. Their realized trajectories need not match once the student policies diverge. Sampled-token OPD applies a clipped surrogate to individual token ratios. Historical Block3 mean instead averages sampled advantages within fixed three-position blocks, exponentiates the sum of valid-token old-current log-probability differences to obtain a joint block ratio, and reduces block losses using fractional valid-token weights. Incomplete blocks are retained. With one response per microbatch, the update averages separately normalized response losses. This changes both credit assignment and ratio/clipping geometry, and does not isolate advantage smoothing alone.

**English, interpretation/limitations.** Under a fixed-horizon, fully on-policy, unclipped idealization, block coupling restores within-block future-credit terms while omitting terms beyond block boundaries. The actual top-p-sampled, masked, frozen-target PPO surrogate is not generally an exact sequence-KL objective. A local averaging argument can improve conditional scalar MSE only when covariance reduction exceeds signal-heterogeneity bias; it gives no universal guarantee for the implemented parameter-gradient variance, optimization stability, or task performance. Relative to token supervision, full blocks can introduce an approximately threefold coefficient scale even before PPO clipping.

**中文，方法。** 两种处理都使用教师对各自学生生成轨迹的每个 token 进行评分；对给定轨迹做 block 分组并不减少教师调用次数。学生策略分化后，两组实际生成的轨迹不必相同。Sampled-token OPD 对逐 token ratio 使用裁剪代理目标；历史 Block3 mean 则对固定三个位置内的 sampled advantage 求均值，将有效 token 的新旧学生 log-probability 差求和后指数化，形成 joint block ratio，再以有效 token 数占 block 宽度的比例归约 loss。短尾 block 保留。在每个 microbatch 只有一个回答的设置下，更新平均的是各回答分别归一化后的 loss。因此它同时改变信用分配、ratio/clip 和 reduction，不能被表述为只做 advantage 平滑。

**中文，解释与限制。** 在固定长度、真正 on-policy 且不裁剪的理想条件下，block 耦合恢复块内未来信用项，但遗漏跨块边界的未来项。实际采用 top-p 采样、mask、冻结目标及 PPO 裁剪的训练代理目标不能泛称为精确序列 KL。局部平均只有在协方差收益超过信号异质性偏差时才改善条件标量 MSE；这不保证实际参数梯度方差、训练稳定性或下游效果普遍改善。即使尚未触发裁剪，完整 Block3 也可能相对 token 监督引入约三倍的系数尺度。

**Related-work framing.** Position this as a study of a particular sampled-OPD joint-action surrogate and its implementation-sensitive tradeoffs. Contrast it with detached old-current drift gates (BPDG), restricted teacher-support matching (Revisiting), span filtering by independent reasoning progress (R2-OPD), and length-normalized sequence ratios (GSPO). GKD/MiniLLM establish the on-policy distillation background. Do not attribute every nearby implementation choice to this work.

## 10. Reading Ledger and Reproducibility

Pages below are PDF page numbers, not claims of visual inspection. PDF text was extracted from downloaded bytes via pdftotext, with no PDF/text files retained. Downloading all bytes does not mean reading every page. Screenshot requests did not yield images inspected by this worker; equations were checked from text and independently against code.

| Key/source | Audited version; total PDF pages where known | Actually read |
|---|---|---|
| gkd | 2306.13649v3; 18 | Abstract/metadata; PDF pp.1-5, emphasis §2-3.2 and Algorithm 1. |
| minillm | 2306.08543v6; 23 | Abstract/metadata; pp.3-5, §2.1-2.3/Algorithm 1. Not full experiments/appendix. |
| revisiting | 2603.25562v2; 26 | Abstract/metadata; pp.2-6 and 14-16. Appendix D read in full; E.1-2 toy details read. Not entire paper. |
| rethinking | 2604.13016v2; 30 | Abstract/metadata; pp.2-5 and 9-16. Not pp.6-8 or all appendices. |
| bpdg | 2606.24084v1; 8 | Entire eight-page PDF text, including references. Referenced works were not all independently read. |
| gspo | 2507.18071v2; 7 | Abstract/metadata; pp.2-4, §2-4.3. |
| r2opd | 2608.19408v1; 20 | Abstract/metadata; pp.1-5, including Proposition 2 and Eq.13. |
| SOD | 2605.07725v3; 32 | Abstract/metadata; pp.3-5. |
| FiRe-OPD | 2606.02684v2; 12 | Abstract/metadata; pp.3-5. |
| TIP | 2604.14084v4; 20 | Abstract/metadata; pp.3-4, including start of §5; not full results. |
| SG-OPD | 2606.09304v1; 14 | Abstract/metadata; pp.3-4. |
| ppo | 1707.06347v2; 12 | Abstract/metadata; pp.1-5, emphasis §2.1, §3, Algorithm 1. |
| dagger | arXiv 1011.0686; AISTATS 2011 | Publisher abstract and official BibTeX only. No PDF/theorem audit. |
| dapo | 2503.14476v2; 16 | Abstract/metadata; pp.3-6, §2.2-3.4. |
| qwen3 | 2505.09388v1; 35 | Abstract/metadata; pp.11-15. |
| llama3 | 2407.21783, current metadata | Abstract/metadata only; no PDF read. |
| llama32 / public Qwen GRPO card | Revisions in §7.1 | Model-information/training-description portions and repository API metadata; not entire licenses or local model weights. |
| math | 2103.03874v2 | Abstract/metadata; PDF pp.3-5; author-repository README. |
| prm800k / MATH-500 | 2305.20050v1; linked repository/card | Paper abstract/metadata only; repository MATH Splits/Answer Grading sections; full short H4 card and viewer row count. |
| MAA organizer | Web page accessed 2026-09-21 | Competition identity/descriptions, not year-specific ML datasets or exam PDFs. |

Complete PDF byte downloads used for the five core OPD texts were SHA-256 checked in memory; version banners matched the audited versions:

| PDF | SHA-256 |
|---|---|
| 2306.13649v3 | a81c71fa0cd15506642092790d854347f030c29ee1d6aa643b55aa51348e6869 |
| 2306.08543v6 | a96fbfb8fcb935aead8dd04d8e375bc12e149a48d5dd26c06e6f9e561c579fc6 |
| 2603.25562v2 | 975135cabbd0cb164f295e8f24b31c0da22ff4210ab713d5825cae2def1fd25d |
| 2604.13016v2 | 7b43c679b5179aade605a2d9636c6a03e49234e0561bb5245e4de4ca6fcee53a |
| 2606.24084v1 | f2d658022b0958b42366a65e9b5b899fe1663eca37e02c943165fec3ebf4685d |

Verification used the existing CPU-capable environment, no package installation:

~~~bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /home/tyf/miniconda3/envs/vllm/bin/python -B -m pytest \
  tests/test_block_supervision.py \
  tests/test_revisiting_opd_block_policy_loss.py \
  tests/test_opd_diagnostics.py -q -p no:cacheprovider
~~~

Result: 31 tests passed. The default Python environment failed torch import with an iJIT_NotifyEvent linkage error; it was not modified. Separate CPU checks used the actual masked_mean AST, autograd, finite differences, and eight-path enumeration as recorded above. These verify local formulas/branches, not GPU execution, distributed weighting for every configuration, or historical training causality.

Final checks: the 31 tests passed again in 2.08 seconds. A limited structured reader checked all 15 braced BibTeX entries for balanced structure, unique keys/fields, required author/title/year/URL fields, arXiv/DOI consistency, and resolution of all currently present manuscript citation keys. It is not a full BibTeX engine; no bibtex executable was on PATH, and no manuscript build was attempted. No whitespace errors were reported for either authored file. The four reviewed manuscript-file hashes remained unchanged at the final read.

The following minimal in-memory checker reproduces the denominator/gradient audit without importing the distributed training stack or writing test files:

~~~python
import ast
import runpy
from pathlib import Path
import torch

ns = runpy.run_path("tests/test_revisiting_opd_block_policy_loss.py")
path = Path("external/revisiting_opd/verl/utils/torch_functional.py")
node = next(n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "masked_mean")
scope = {}
exec(compile(ast.Module(body=[node], type_ignores=[]),
             "production_masked_mean", "exec"), scope)
ns["core_algos"].verl_F.masked_mean = scope["masked_mean"]
loss_fn = ns["compute_policy_loss"]
for k in (1, 3):
    x = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    loss = loss_fn(torch.zeros_like(x), x, torch.ones_like(x),
                   torch.ones_like(x), cliprange=0.2, opd_block_size=k,
                   opd_block_advantage_mode="mean")[0]
    print(k, loss.item(), torch.autograd.grad(loss, x)[0].tolist())
~~~

Open gaps are explicit: no independently verified full-paper reading beyond BPDG; no exact local dataset/model-hash certification; no new empirical baseline or repeated-seed experiment; no exhaustive novelty search; no main-manuscript build performed by this worker.

## 11. Read-Only Review of the Current Draft

This review covers method_en.tex, method_zh.tex and relevant related-work/protocol text in content_en.tex/content_zh.tex as read after the author's fractional-weight correction. No manuscript files were edited. Method snapshots: English SHA-256 8926358c163cdf56d4231d9dd31cd2f88ac05344e227e7d533027f4581dfca1f; Chinese acbae6dcb4064e6a0ebdefffda512fe1cfb1bbe27f46138399a263950ce4461d.

**Substantive finding:** content_en.tex:27 and content_zh.tex:46-48 describe BPDG as applying stale-policy corrections and suggest a parallel change in probability-ratio granularity. This can misidentify the comparator. Its actual operation is detached, mean-normalized old-current **drift gating of unchanged token losses**, not a new joint block PPO ratio or a demonstrated off-policy correction. Suggested factual replacement: "BPDG aggregates sampled old-current log-probability shifts to construct detached loss weights, while retaining the underlying token-level objective; our method instead aggregates teacher feedback and changes the PPO ratio itself." See BPDG §3, PDF pp.3-4 and §7 p.6. This is a literature-identity correction, not a claim that the draft's block-loss algebra is wrong.

**No substantive error found in the revised method equations:** fractional n_j/k weighting and the eta-free n_j/T form agree; holes are not packed; the equal-feedback proposition is correct under its full-block/ratio-one/no-clip assumptions and explicit eta omission; cross-credit is not equated with task error; the noise-MSE paragraph concerns a hypothetical token-ratio objective, not demonstrated Block3 gains.

At the review snapshot, appendix_en.tex and appendix_zh.tex did not yet exist in this directory. The appendix dual-clip formula and formal proof therefore could not be checked. This is a reading limitation, not a claim that an unfinished draft failed a build.

### 11.1 Dual Clip in the Manuscript's Maximization Convention

Section 2 used a **minimized** loss. The manuscript uses a **maximized** surrogate C and then minimizes -C; translating signs matters. Its correct formula is

\[
 {\cal C}_0(r,a)=\min\{ra,\operatorname{clip}(r,.8,1.2)a\},\qquad
 {\cal C}(r,a)=
 \begin{cases}
   \max\{{\cal C}_0(r,a),3a\},&a<0,\\
   {\cal C}_0(r,a),&a\ge0 .
 \end{cases} \tag{M17}
\]

The loss is -C. Using min instead of max in the negative-advantage branch of this maximization convention would be a genuine sign error. The general settings require c>1 for the code and 0<epsilon_low, epsilon_high for an open inactive neighborhood of r=1; here c=3 and both epsilons=.2. The 1.1^3 example establishes a different clipping region; zero gradient there specifically requires a positive block advantage. It does not imply saturation of every negative-advantage example.

### 11.2 Formal Proposition and Proof for an Appendix

**Proposition (within-block future credit under ideal sampling).** Fix a prompt, a deterministic horizon T, and a deterministic contiguous partition of positions 1,...,T. There are no holes or data-dependent boundaries. Let q be a fixed autoregressive teacher positive on student support. Assume student distributions have locally parameter-independent support, differentiable log-probabilities, and sufficient integrability to interchange differentiation and expectation. Draw Y from P_theta0, not a nucleus-truncated sampler. Set
\(c_t=\log p_{\theta_0}(Y_t|H_t)-\log q(Y_t|H_t)\) and
\(s_t=\nabla_\theta\log p_\theta(Y_t|H_t)|_{\theta_0}\).
Let e(t) be the deterministic last position of t's block. With frozen A_t=-c_t, inactive clipping, and eta=0, the expected historical block-loss gradient equals

\[
 {1\over T}\mathbb E\left[\sum_{t=1}^Ts_t
                                      \sum_{j=t}^{e(t)}c_j\right]. \tag{M18}
\]

Its bias relative to the sequence reverse-KL gradient/T is

\[
 \mathbb E[g_{\rm blk}]
 -{1\over T}\nabla_\theta D_{\rm KL}(P_\theta\Vert Q)|_{\theta_0}
 =-{1\over T}\mathbb E\left[\sum_{t=1}^Ts_t
                                          \sum_{j>e(t)}c_j\right]. \tag{M19}
\]

**Proof.** For each fixed history, differentiating normalization gives
\(\mathbb E[s_t|H_t]=\sum_vp_{\theta_0}(v|H_t)
\nabla_\theta\log p_\theta(v|H_t)|_{\theta_0}=0\).
For j<t, c_j is H_t-measurable, hence
\(\mathbb E[c_js_t]=\mathbb E[c_j\mathbb E[s_t|H_t]]=0\).
The pathwise gradient in M7 with A=-c expands into all pairs within each block; eliminating the j<t terms in expectation leaves M18. Separately, differentiating
\(\sum_YP_\theta(Y)\log[P_\theta(Y)/Q(Y)]\)
gives \(\mathbb E[(\sum_jc_j)(\sum_ts_t)]+\mathbb E[\sum_ts_t]\).
The second expectation is zero by normalization. Removing past terms from the first gives
\(\mathbb E[\sum_ts_t\sum_{j\ge t}c_j]\).
Subtracting this expression/T from M18 gives M19. No independence between c_j and s_t for future j is assumed. End of proof.

For k=1 the estimator retains only immediate terms. A single full-sequence block recovers the full fixed-horizon gradient/T under these assumptions. General blocks restore some potentially useful future credit while leaving an omitted-boundary term. Zero-expectation past terms can still affect variance. The sign/norm of the omitted vector, and downstream usefulness of restored terms, are not determined by this proposition. In particular, the term "leakage" must not classify all within-block future credit as erroneous.

**Boundary of applicability.** Top-p=.9 behavior, EOS-dependent lengths and denominators, action-dependent masks/boundaries, stale parameters away from theta_0, and active ordinary/dual clipping are not covered. Fractional tail weighting remains algebraically correct for a deterministic final short block, but a randomly stopped response is not automatically such a partition.

**MSE assumption to state in the appendix.** The draft's matrix identity is valid for deterministic W and mu, or conditionally on F with W, mu fixed, E[epsilon|F]=0, and conditional covariance Sigma_F. A data-dependent W selected from the same noisy advantages cannot simply be pulled outside expectation. Fixed block and fixed sliding-window averaging meet the deterministic-W requirement; no claim about adaptive selection or actual gradient variance follows. This qualification completes the intended signal-estimation argument without turning it into a Block3 performance theorem.
