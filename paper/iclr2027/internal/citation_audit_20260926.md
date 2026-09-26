# 参考文献真实性与引用对应核验

核验日期：2026-09-26。
核验对象：当前中英文论文共同使用的 `references.bib`，以及主文和附录的引用位置。
核验时仓库版本：`7470209`。
BibTeX SHA-256：`98a4cb67d60d452d58036562ff4a3bc23b3f13e87f29b340acb003bdcf5c6d6b`。

## 结论与范围

- 共18条：16条arXiv论文、1条PMLR会议论文、1条Meta官方模型卡，全部找到对应的第一方来源。
- 17篇论文的题名、所列作者顺序和年份均与官方页面元数据匹配；模型卡的发布者、模型身份和发布日期人工核对通过。
- BibTeX结构解析：重复key为0，正文及附录未定义的引用key为0，未使用条目为0；编译后的英文参考文献也包含18条。
- 未发现虚构条目或编号指向其他论文的情况。对本文实际引用的主要方法与来源陈述，未发现相反或不支持的原文证据。
- 本次是文献身份、元数据与被引用段落的审查，不等于独立复现这些论文，也不表示重新逐页精读全部17篇论文。全文技术核查集中于本文引用的章节、公式和模型/数据来源。
- 没有修改正文、参考文献条目、PDF、实验结果、训练代码或远端队列。此文件是内部核验记录，不加入匿名投稿源码包。

## 逐条身份核验

所有arXiv年份均按首次公开年份核对，版本日期与年份不是同一字段。作者姓名比较统一姓/名顺序、空格、标点和重音；Llama 3明确使用前三位作者加others，不将缩写作者表误当成完整作者表。

| Key | 文献 | 官方来源 | 核验结果 |
|---|---|---|---|
| `opsd` | Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models | [2601.18734v3](https://arxiv.org/abs/2601.18734v3) | 2026，题名及7位作者匹配，v3为2026-03-20 |
| `gkd` | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes | [2306.13649v3](https://arxiv.org/abs/2306.13649v3) | 首发2023，7位作者匹配，页面注明ICLR 2024 |
| `minillm` | MiniLLM: On-Policy Distillation of Large Language Models | [2306.08543v6](https://arxiv.org/abs/2306.08543v6) | 首发2023，4位作者匹配，题名与指定v6一致；页面注明ICLR 2024 |
| `revisiting` | Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes | [2603.25562v2](https://arxiv.org/abs/2603.25562v2) | 2026，题名及7位作者匹配 |
| `rethinking` | Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe | [2604.13016v2](https://arxiv.org/abs/2604.13016v2) | 2026，题名及11位作者匹配 |
| `bpdg` | Blockwise Policy-Drift Gating for On-Policy Distillation | [2606.24084v1](https://arxiv.org/abs/2606.24084v1) | 2026，Liwen Zheng与Haiyun Jiang，匹配 |
| `ppo` | Proximal Policy Optimization Algorithms | [1707.06347v2](https://arxiv.org/abs/1707.06347v2) | 2017，题名及5位作者匹配 |
| `dapo` | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | [2503.14476v2](https://arxiv.org/abs/2503.14476v2) | 2025，题名及35位作者匹配 |
| `qwen3` | Qwen3 Technical Report | [2505.09388v1](https://arxiv.org/abs/2505.09388v1) | 2025，题名及60位作者匹配 |
| `llama3` | The Llama 3 Herd of Models | [2407.21783](https://arxiv.org/abs/2407.21783) | 2024，当前为v3；所列前三位作者匹配，后续作者按others省略 |
| `r2opd` | Beyond Imitation: Filtering On-Policy Distillation by Reasoning Progress | [2608.19408v1](https://arxiv.org/abs/2608.19408v1) | 2026，题名及5位作者匹配 |
| `gspo` | Group Sequence Policy Optimization | [2507.18071v2](https://arxiv.org/abs/2507.18071v2) | 2025，题名及12位作者匹配 |
| `dagger` | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning | [PMLR官方条目](https://proceedings.mlr.press/v15/ross11a.html) | 2011，3位作者、会议、卷15、页627--635及编辑信息匹配 |
| `llama32` | Llama 3.2 1B Instruct: Model Card | [Meta官方模型卡](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Meta，发布日2024-09-25，说明同系列1B/3B预训练及指令模型 |
| `math` | Measuring Mathematical Problem Solving With the MATH Dataset | [2103.03874v2](https://arxiv.org/abs/2103.03874v2) | 2021，题名及8位作者匹配，页面注明NeurIPS 2021 |
| `prm800k` | Let's Verify Step by Step | [2305.20050v1](https://arxiv.org/abs/2305.20050v1) | 2023，题名及10位作者匹配 |
| `fireopd` | Filter, Then Reweight: Rethinking Optimization Granularity in On-Policy Distillation | [2606.02684v2](https://arxiv.org/abs/2606.02684v2) | 2026，题名及9位作者匹配 |
| `tip` | TIP: Token Importance in On-Policy Distillation | [2604.14084v4](https://arxiv.org/abs/2604.14084v4) | 2026，题名及6位作者匹配 |

## 本文陈述与来源对应

| 本文引用用途 | 第一方证据与检查结果 |
|---|---|
| GKD允许学生生成前缀和不同散度，而非on-policy必然等于reverse KL | [GKD第3节](https://arxiv.org/html/2306.13649v3)明确将序列来源和散度选择分别作为自由度；与本文背景段一致。 |
| MiniLLM已有未来奖励信用分配 | [MiniLLM第2.2节](https://arxiv.org/html/2306.08543v6)包含单步/长期梯度分解与后续奖励项；本文没有把时间信用分配本身声称为首次。 |
| sampled-token基线来源于Revisiting OPD，而不是将其Top-K新方法冒充基线 | [Revisiting第2.1--3节](https://arxiv.org/html/2603.25562v2)区分sampled-token近似与teacher-top-K方法；本地`.gitmodules`及upstream README指向作者的`hhh675597/revisiting_opd`仓库。 |
| OPSD使用同一模型、不同上下文构造自教师 | [OPSD第3.2节](https://arxiv.org/html/2601.18734v3)定义共享参数的两个条件分布及学生rollout监督；本文保留独立教师，没有混写成OPSD实现。 |
| Rethinking OPD是熵、overlap及师生兼容性诊断的来源 | [Rethinking第2.3及3节](https://arxiv.org/html/2604.13016v2)包含对应定义与实验；本文附录另写明自身实现公式。 |
| Qwen3-4B-Base-GRPO由Rethinking OPD配套发布 | [官方模型卡](https://huggingface.co/Thinking-Space/Qwen3-4B-Base-GRPO)明确关联2604.13016，列明Qwen3-4B-Base、GRPO和DAPO-Math-17k-Processed；不是根据名字推断。 |
| Blockwise Policy-Drift Gating使用old/current漂移生成停止梯度、均值归一化的权重 | [BPDG第3节](https://arxiv.org/html/2606.24084v1)定义该机制，并保留原teacher target与位置损失；本文与自身joint-ratio方法的区分有依据。 |
| GSPO采用长度归一化的序列重要性比率 | [GSPO第4.1节公式7](https://arxiv.org/html/2507.18071v2)明确为序列ratio的长度次方根，与本文block乘积的区别正确。 |
| DAPO中的全局token加权不同于逐response归一化 | [DAPO第3.3节](https://arxiv.org/html/2503.14476v2)解释长样本对整体梯度的作用；本文没有因配置名token-mean而声称完全相同。 |
| FiRe-OPD、TIP、R2-OPD分别进行轨迹/token/进展信号选择 | [FiRe-OPD摘要](https://arxiv.org/abs/2606.02684v2)、[TIP摘要](https://arxiv.org/abs/2604.14084v4)、[R2-OPD原文](https://arxiv.org/html/2608.19408v1)与本文简短描述一致；此处未独立复核它们报告的数值增益。 |
| MATH500的PRM800K划分谱系 | [OpenAI PRM800K官方仓库的MATH Splits及Citation](https://github.com/openai/prm800k)解释保留500题及对应论文；本文不是声称原MATH论文本来只有500题。 |
| Llama 3.2的1B/3B模型身份 | 直接依据[Meta 3.2模型卡](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)，而非只依赖Llama 3技术报告；论文已有独立`llama32`引用。 |

## 容易被误判的情况

1. GKD和MiniLLM写2023，但说明ICLR 2024，是预印本首发年与会议年不同，不是虚构年份。
2. MiniLLM当前所引v6的题名含On-Policy；不能拿早期题名不同就判定假引用。
3. `llama3`目前未固定v3 URL且缩略作者表；现有内容核验通过，但若以后更新该来源，应重新核对版本，不将该条目单独当作Llama 3.2的小模型技术来源。
4. 文献真实存在不代表它证明了我们的实验结论；Block3的性能与机制主张仍须由本项目自己的数据、推导及消融支撑。

## 检查方式

第一轮通过浏览工具访问arXiv、PMLR、作者/机构模型卡及官方仓库；第二轮使用BibTeX解析器读取18条记录，并从17篇论文官方页面提取`citation_title`、`citation_author`及年份进行比对。题名、作者顺序和年份17/17匹配。另检查所有根目录LaTeX文件中的引用key，以及现有编译产物的18个bibliography条目。

本次未新增或替换参考文献，因此无需重新编译或替换已交付PDF。后续如改动参考文献，应以本文件记录的BibTeX哈希判断此核验是否仍覆盖当前版本。
