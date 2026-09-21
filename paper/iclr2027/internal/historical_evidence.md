# 历史实验证据审计

> 内部全量账本，不等同论文主表。审计快照：2026-09-21 00:53:57（北京时间；UTC 2026-09-20T16:53:57）。
> 分工：本线程只负责历史审计。新 Token 评测由父线程处理；本文件不声称其已启动或完成。

## 审计边界

只读本地 `docs/results/`、`results/`、`reports/`、`configs/`、`docs/plans/`；只对 `ml2` 做已有 JSON/清单查证，并补充CPU只读数据列扫描（仅回传统计，footer初探两行样本未写入产出），没有连接 `train`，没有 GPU 操作，没有运行训练或 grader，没有改实验代码，也不提交。仅本文件及同目录 `historical_evidence.json` 是本任务产出。

优先级：已最终化的每次评测 `acceptance.json`/`summary.json` > 已验证哈希的配对 JSON > curated JSON > MD/HTML 舍入表 > 计划。不能仅凭文件日期择新，也不能把同权重新生成的评测覆盖旧结果。每一个数值在 JSON 中带来源及字段路径；来源索引给出原路径，完整 SHA256 在 JSON 的 `sources`。

可信度 A：本次实际读到 ml2 原始结果/验收 JSON、小型modelcard/metadata，或完成原数据列统计；小文件计算哈希，完整训练Parquet仅保留历史manifest哈希，没有重读所有逐题 raw/graded；B：本地最终化配对 JSON 与 `finalization_hashes.sha256` 匹配，未连接 train；C：curated JSON；D：仅文档/HTML显示值；P：计划/配置，不是执行证明。A 级 `passed=true` 只表示既有验收记录，本次不冒充重新完整判分。

## 重要发现

- 历史不能写成普适正面结果：Qwen1.7 同机 Step200 外部 grader 四任务有正向差异，但 DeepSeek/JustRL 没满足原复现门槛；0.6B 非 thinking 与 Llama historical-prompt 的 Block3 均有明显负面结果。
- Block10 双机复跑都在 Step200 四任务零分；它们是同一个训练 seed 的两机运行，不是多 seed。无 NaN/OOM 不等于输出健康。
- 旧文档的“Sliding3评测中”“0.6B Block3未评”“Llama Token训练中”“Qwen4只评完Step200”已过时；本次读到的最终 JSON 已补齐相关结果。
- 新 Qwen4 completion Block3 四个 checkpoint 已全评；新 Token 200步训练已完成，但本账本无其完整 benchmark 分数，匹配的 completion Base 分数也缺。当前只能报 Block3 绝对分数，不能宣称增益。
- 早期 `Qwen/Qwen3-0.6B`/`Qwen/Qwen3-4B` 不是后续 Base/公共GRPO师生。`Qwen4` 是项目的4B简称，实际为 `Qwen3-4B-Base`，不是Qwen4代。
- 历史请求固定 seed21 导致同题名义8条训练输出可完全重复；不应当作8个独立样本。新completion同时改prompt和request-seed，不是单因素消融。

## 协议与呈现

早期 clean-room 使用3400 prompt：GSM8K train 2000，加 Hendrycks MATH 七子类各200。full eval 为 GSM8K1319/MATH500500，n1，temperature0.7/top_p0.95，最大256输出token；100题 pilot 单列，不冒充 full。早期 online simple grader 与事后 primary regrade 单列，后者不是后续 SHA04f7 外部 grader。

后续 DAPO 共同预算：历史展开pool 1,791,700行（精确唯一题面17,237；题答组合17,249，补查见下），seed21，200steps，4prompt x 8名义rollout，LR2e-6，mini32/PPO epoch1，prompt2048/response16384，temperature1/top_p0.9。每次full eval为 MATH500500、AIME24 30、AIME25 30、AMC23 83，每题n8，共5144响应；除旧train sweep未显式设seed外，已固定评测seeds21..28。数据和grader完整hash见JSON `protocols.dapo_common`。

提示不能混：历史Qwen训练ChatML+think要求/kwargs={}，评测ChatML+显式false；0.6B新nonthinking仍是ChatML，但两端统一false并同步EOS/mask修正；Llama始终为原生chat/BOS/EOS，historical只改变训练正文think要求；新Qwen completion才完全绕过ChatML并采用独立派生request seeds。Llama的 `enable_thinking` 不是其原生模板的模式开关。

表格数值单位为百分比，n8单元按 **Avg@8 / Pass@8**，舍入至两位；n1表按准确率。pilot100保留 **mean_score / pass_at_k**，k未登记，不伪称Avg@8。JSON保留原精度及不同grader字段。`缺失`不是0；没有MATH/AMC列值也不代表零。每个行ID和S来源对应JSON，多个来源表示不同benchmark的各自来源，不是平均。

## 匿名复现来源补查

补查时间：2026-09-21 01:14:32（北京时间，为训练池完整列扫描完成时间；不改变前述评测快照）。此处区分公开来源身份、历史命令登记和本次实际数据观测，未知项不补猜。

### DAPO训练池

- **行数与unique已实查：** 对 `train.parquet` 的题面、答案、index、data_source 做CPU流式完整扫描，读取1,791,700行。精确 `env_kwargs.question` 字符串有 **17,237** 种，精确 `(question, ground_truth)` 有 **17,249** 种；不做额外去空格、数学等价或语义去重。[S264](#s264)。
- **展开规则是实际观测：** question/answer序列等于前 **17,917行按顺序重复100次**；`extra_info.index` 连续从0到1,791,699。17,917也不是唯一题数。题面出现次数100/200/300/400的题目数量分别为16,592/617/21/7；题答组合分别16,616/605/21/7。只核了投影列，不能说完整物理行完全重复。
- **公开源与断链：** 保留转换器的默认源是 `BytedTsinghua-SIA/DAPO-Math-17k` 的 `data/dapo-math-17k.parquet`。实际历史manifest却记录 `copied_train_source_parquet`、URL=null，源是更早 `math_opd_dapo17k_hf_full/train.parquet`，源/目标登记SHA均为 `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`。原下载Hub revision、原始未转换文件哈希、谁在何处创建100倍展开及其命令均缺失；本次不重新散列1.6GB文件。[S240](#s240)、[S241](#s241)。
- **当前转换器不能冒充历史命令：** 它清理外层解题/末行格式指令，默认按题答组合去重，有 `--keep-train-duplicates`，没有显式repeat100；而旧manifest没有当前版本的去重统计。直接跑现在默认脚本会改变训练池，不能复现本历史协议。没有执行该脚本。
- **200update预算：** seed21、4prompt/update、每prompt名义8rollout，即800个行位次和6400条名义响应，不保证800道唯一题，也不等于6400条独立响应。JSON `data_provenance.dapo_train` 保留扫描算法、计数、源链和缺失项。

### 四个Benchmark

| benchmark | 行数/精确唯一题面 | 可核实的公开来源 | 版本与证据 |
|---|---:|---|---|
| MATH500 | 500 / 500 | `HuggingFaceH4/MATH-500`，`test` | 原HF revision缺失；转换后JSONL SHA已核对 [S240](#s240)、[S261](#s261) |
| AIME24 | 30 / 30 | `BytedTsinghua-SIA/AIME-2024`，`data/aime-2024.parquet` | 历史URL为resolve/main，commit未固定；原Parquet登记SHA与eval JSONL SHA保留 [S240](#s240)、[S260](#s260) |
| AIME25 | 30 / 30 | `thunlp/OPD`，`datasets/test_data/AIME25/test.parquet` | 仓库checkout文件已追踪且对HEAD无差异，SHA与历史manifest相等；更上游HF namespace未知 [S258](#s258)、[S262](#s262) |
| AMC23 | 83 / 83 | `thunlp/OPD`，`datasets/test_data/AMC23/test.parquet` | 同上；本系列就是83行版本，不换成别的常见AMC题数；更上游HF namespace未知 [S259](#s259)、[S263](#s263) |

OPD公共仓库本地checkout commit为 `1fd6cca846126af90d82ef122e8af261f59d2d37`；JSON保存可用于匿名附录的公开repo/file URL。它不是本项目 `revisiting_opd` 的训练runtime commit。四份转换后JSONL均重读计数并与manifest SHA逐一吻合，未重新生成或重新判分；精确字符串unique不代表语义去重。AIME25/AMC23原Parquet分别为30/83行，AMC23原source字段为 `amc`，不据此虚构更上游来源。

### 公共Teacher

**完整repo ID：`lllyx/Qwen3-4B-Base-GRPO`。** 依据是本地model card的Usage `model_id`，并非在线Hub检索；README SHA256为 `25889d058465bc1ff4372e91148f67569e4e05d27dd87a76d058f6720e7ff248`。model card登记base `Qwen/Qwen3-4B-Base`、GRPO、`DAPO-Math-17k-Processed` 及相关论文2604.13016。这是公共teacher上游训练说明，不能替代学生DAPO池来源证明。[S239](#s239)。

**Revision分层：** 原run-card没有Hub revision，JSON `models.qwen4_grpo.revision` 仍为null。本次独立读取15份本地下载 `*.metadata`，第一行一致为 `1f3b2966edfb75f2f98a00617588c1f748088422`；两片safetensors ETag与既有资产SHA清单一致。这个值保存在 `local_download_revision_evidence`，属于本地缓存恢复证据，不是从model card推得，也不冒充当时显式pin的命令或在线Hub核验。没有下载新模型、重新读取权重或修改旧run-card。[S251](#s251)、[S252](#s252)。

### 自动制表入口

`evaluations[]` 是逐系列、方法、checkpoint、view的记录；每项 `per_task[benchmark].metrics` 保留原精度，`source_pointer` 和 `source_fields` 可定位 `sources[Sxxx].path` 原始JSON字段。务必用 `series + arm + checkpoint_step + view` 分组，按benchmark各自保留来源；不同grader/prompt/生成批次不可合并。历史D级显示值单独标注舍入，缺失null不补零。统计是171条评价记录（包含按任务分开的记录及缺失记录），不是171次独立训练。

## 系列总览

| 系列 | 状态 | 论文位置建议 |
|---|---|---|
| [早期3400-prompt权重方法与seed7/13 pilot](#early_weights50) | 完成，探索性混合/负面结果 | 历史附录；不是后续 DAPO 主表 |
| [早期200-step outcome reweight与standard/topk曲线](#early_outcome200) | 完成，晚期退化；重权未可靠消除退化 | 历史/负面附录 |
| [早期100-step reference-anchor对照](#early_anchor100) | 完成，效果依任务/对照而异 | 历史附录 |
| [早期seed7 naive block k1/2/3](#early_blocksum50) | 完成，k3微弱方向收益且梯度放大，k2负面 | 历史附录；舍入数值不能作为高精度主表 |
| [早期seed7 Block3 sum/mean/mixed权重](#early_blockadv50) | 完成，mean微弱正向；mixed在MATH500负面 | 历史归一化附录 |
| [Qwen1.7 Base blocksize sweep k1/3/5/10](#qwen17_sweep) | 完成：k3混合、小幅；k5回退；k10四任务零分 | 历史尺寸消融，标明跨机器混杂 |
| [Block10双机独立诊断复跑](#block10_collapse) | 两机完成但均发生科学失败；Step200全零 | 必须保留的负面/稳定性附录 |
| [Qwen1.7 Base 同机Token/Block3严格配对](#qwen17_ml2) | 完成，Step200四任务单seed正向；Step100存在负面 | 历史受控配对可入历史主表，不能外推新completion协议 |
| [Qwen1.7历史六权重新生成评测](#qwen17_reeval) | 六项完成；非新训练，非新增seed | 重生成敏感性/可追溯附录 |
| [DeepSeek-R1-Distill/JustRL配对](#deepseek_justrl) | 完成；未达到预定跨师生方向复现 | 必须保留的跨模型负面对照 |
| [Qwen1.7 random3/sliding3](#window_random_sliding) | 两臂及50/100/200评测完成；sliding相对random后期正向，但两者退化 | 窗口方法对照附录；不能冒充fixed3对照 |
| [Qwen0.6 Base旧thinking-unspecified提示](#qwen06_old_prompt) | Token训练完成；Step50评测用户中止；Block3未启动 | 历史中止记录；无完整配对分数 |
| [Qwen0.6 Base显式非thinking配对](#qwen06_nonthinking) | Token/Block3训练与全评完成；Token提升、Block3严重负面 | 必须保留的同协议小模型负面对照 |
| [Llama3.2原生非thinking提示尝试](#llama_native_stopped) | Base评测完成；Token在Step146被用户停止；Block3未启动 | 历史中止记录，不构成方法对照 |
| [Llama3.2 historical17训推差异配对](#llama_historical) | 两组200步及全部评测完成；Block3明显负面 | 必须保留的跨模型负面对照 |
| [Qwen4旧ChatML Block3-first失败/中止](#qwen4_chatml_stopped) | 正式Step4用户中止；无正式checkpoint/eval | 协议失败与负面诊断附录 |
| [Qwen4新completion Block3与新Token](#qwen4_completion) | Block3四checkpoint全评完成；Token200步训练已完，benchmark等待 | 候选新协议主表，匹配Base/Token评测未齐前不可写增益 |

<a id="early_weights50"></a>
## 早期3400-prompt权重方法与seed7/13 pilot

**状态：** 完成，探索性混合/负面结果。**用途：** 历史附录；不是后续 DAPO 主表。

**师生身份：** 学生 `Qwen/Qwen3-0.6B`，revision未记录；教师 `Qwen/Qwen3-4B`，revision未记录。
**训练seed：** 7, 13；训练步数 50。协议：`early_cleanroom`。

- 历史run：`/mnt/data/cpfs/Yaleon/opd_cleanroom_20260707/runs/pilot50_20260707_230404`。
- 历史run：`/mnt/data/cpfs/Yaleon/opd_cleanroom_20260707/runs/ablation50_20260708_003315`。
- pilot50: standard、topk_router、adaptive_exopd、routed_adaptive 各50步；ablation50: seed13_standard、seed13_topk_default、seed7_topk_strict、seed7_routed_lam110。
- 100题 pilot 与 full GSM8K1319/MATH500500 独立分层；不能把100题分数当 full。mean_score/pass_at_k 原字段保留，100题摘要未记k，不伪标 Avg@8。
- seed7 standard/topk 的 full regrade MATH500=0.202/0.194；routed_lam110 GSM8K=0.4723275208491281，仍是选择后单seed探索。seed13只覆盖standard/topk小样本，不是所有方法多seed。
- 记录中的 topk_router 是本地路由权重变体，不能自动改名为已实现的论文 Student-TopK/Teacher-TopK/FiRe/TOPD；准确公式/阈值缺失。
- 已记录 pilot LR5e-7；50-step training summary 有长度/梯度/lambda/weight，adaptive_exopd 最后一步lambda1.25不是全程固定系数证明。

**缺失与混淆：**

- 早期 model Hub revisions、训练命令/各权重公式参数与完整采样配置未在此次允许来源核齐；不能从后续block配置追填。
- pilot100 eval 的k/seed未在该摘要登记；full已知n1协议不能自动套用。
- adaptive_exopd/routed_adaptive 的 full 已完成分数未找到；不是零。

### 100题pilot原始评分（k未登记）

| 方法/对象 | Step | GSM8K | MATH500 | 记录/来源 |
|---|---:|---:|---:|---|
| base | 0 | 43.00 / 43.00 | 22.00 / 22.00 | E001, E006；[S064](#s064), [S021](#s021) |
| standard | 50 | 46.00 / 46.00 | 25.00 / 25.00 | E002, E007；[S064](#s064), [S021](#s021) |
| topk_router | 50 | 44.00 / 44.00 | 25.00 / 25.00 | E003, E008；[S064](#s064), [S021](#s021) |
| adaptive_exopd | 50 | 39.00 / 39.00 | 22.00 / 22.00 | E004, E009；[S064](#s064), [S021](#s021) |
| routed_adaptive | 50 | 40.00 / 40.00 | 26.00 / 26.00 | E005, E010；[S064](#s064), [S021](#s021) |
| seed13_standard | 50 | 43.00 / 43.00 | 22.00 / 22.00 | E011, E015；[S065](#s065), [S066](#s066) |
| seed13_topk_default | 50 | 38.00 / 38.00 | 27.00 / 27.00 | E012, E016；[S065](#s065), [S066](#s066) |
| seed7_topk_strict | 50 | 41.00 / 41.00 | 20.00 / 20.00 | E013, E017；[S065](#s065), [S066](#s066) |
| seed7_routed_lam110 | 50 | 44.00 / 44.00 | 25.00 / 25.00 | E014, E018；[S065](#s065), [S066](#s066) |

### full n1：原始simple grader

| 方法/对象 | Step | GSM8K | MATH500 | 记录/来源 |
|---|---:|---:|---:|---|
| base | 0 | 41.70 | 16.00 | E019, E035；[S022](#s022) |
| routed_lam110 | 50 | 41.93 | 16.00 | E021, E037；[S022](#s022) |
| standard | 50 | 41.77 | 16.80 | E023, E039；[S022](#s022) |
| topk_router | 50 | 41.70 | 16.40 | E025, E041；[S022](#s022) |

### full n1：事后primary regrade

| 方法/对象 | Step | GSM8K | MATH500 | 记录/来源 |
|---|---:|---:|---:|---|
| base | 0 | 46.25 | 20.20 | E020, E036；[S022](#s022) |
| routed_lam110 | 50 | 47.23 | 20.20 | E022, E038；[S022](#s022) |
| standard | 50 | 46.17 | 20.20 | E024, E040；[S022](#s022) |
| topk_router | 50 | 46.17 | 19.40 | E026, E042；[S022](#s022) |

**系列证据：** [S019](#s019)、[S020](#s020)、[S021](#s021)、[S022](#s022)。

<a id="early_outcome200"></a>
## 早期200-step outcome reweight与standard/topk曲线

**状态：** 完成，晚期退化；重权未可靠消除退化。**用途：** 历史/负面附录。

**师生身份：** 学生 `Qwen/Qwen3-0.6B`，revision未记录；教师 `Qwen/Qwen3-4B`，revision未记录。
**训练seed：** 7；训练步数 200。协议：`early_cleanroom`。

- 历史run：`/mnt/data/cpfs/Yaleon/opd_cleanroom_20260707/runs/outcome200_20260708_020543`。
- 200步正式组 seed7_standard、seed7_topk_router、seed7_outcome_reweight、seed7_outcome_topk_router；本地路径和ml2拷贝路径同时保留。
- 同名 standard_step100/150 来自 outcome200 的checkpoint，不是 anchor100。同名Step50在总曲线中来自先前 pilot50，不能画成同一200步run的连续已评曲线。
- Step200 full regrade: MATH500 standard0.148、topk0.162、outcome0.166、outcome_topk0.164；相比各自原始Base参考0.202无恢复证据。
- 线上 simple grader 与事后 primary grader 保留为独立视图，primary数字不替换旧原字段；终点原始summary已在ml2只读验证。

**缺失与混淆：**

- 200-step各重权参数/训练响应上限/是否warm start缺原训练命令；平均长度约234不能推为配置值。
- outcome两种变体50/100/150 checkpoint虽声明保留，但未找到已评分数；standard/topk自身Step50分数未找到。

### full n1：原始simple grader

| 方法/对象 | Step | GSM8K | MATH500 | 记录/来源 |
|---|---:|---:|---:|---|
| standard | 100 | 35.25 | 13.80 | E027, E043；[S022](#s022) |
| standard | 150 | 33.43 | 12.40 | E029, E045；[S022](#s022) |
| topk_router | 100 | 35.25 | 14.20 | E031, E047；[S022](#s022) |
| topk_router | 150 | 34.87 | 12.00 | E033, E049；[S022](#s022) |
| outcome_reweight | 200 | 33.36 | 12.60 | E051, E059；[S067](#s067), [S071](#s071) |
| outcome_topk_router | 200 | 33.21 | 12.20 | E053, E061；[S068](#s068), [S072](#s072) |
| standard | 200 | 32.15 | 11.80 | E055, E063；[S069](#s069), [S073](#s073) |
| topk_router | 200 | 33.89 | 13.20 | E057, E065；[S070](#s070), [S074](#s074) |

### full n1：事后primary regrade

| 方法/对象 | Step | GSM8K | MATH500 | 记录/来源 |
|---|---:|---:|---:|---|
| standard | 100 | 39.95 | 16.20 | E028, E044；[S022](#s022) |
| standard | 150 | 38.06 | 15.80 | E030, E046；[S022](#s022) |
| topk_router | 100 | 40.26 | 16.20 | E032, E048；[S022](#s022) |
| topk_router | 150 | 39.27 | 14.40 | E034, E050；[S022](#s022) |
| outcome_reweight | 200 | 37.83 | 16.60 | E052, E060；[S067](#s067), [S071](#s071) |
| outcome_topk_router | 200 | 38.51 | 16.40 | E054, E062；[S068](#s068), [S072](#s072) |
| standard | 200 | 36.16 | 14.80 | E056, E064；[S069](#s069), [S073](#s073) |
| topk_router | 200 | 38.82 | 16.20 | E058, E066；[S070](#s070), [S074](#s074) |

**系列证据：** [S023](#s023)、[S024](#s024)、[S025](#s025)、[S026](#s026)。

<a id="early_anchor100"></a>
## 早期100-step reference-anchor对照

**状态：** 完成，效果依任务/对照而异。**用途：** 历史附录。

**师生身份：** 学生 `Qwen/Qwen3-0.6B`，revision未记录；教师 `Qwen/Qwen3-4B`，revision未记录。
**训练seed：** 未在该系列原始摘要登记；训练步数 100。协议：`early_cleanroom`。

- 历史run：`/mnt/data/cpfs/Yaleon/opd_cleanroom_20260707/runs/anchor100_20260708_044116`。
- standard、topk_router、anchored_standard、anchored_topk_router各100步，在独立anchor100目录训练；不与outcome200的Step100合并。
- primary: anchored_standard对standard在GSM8K和MATH500均较低；anchored_topk对topk两项略高。只有单次记录，不证明普适有效。
- training summary记录reference_gap及positive-reference-gap-rate；这些观测不能反推出reference损失的公式/系数。
- 该师生归属依clean-room目录历史说明；本次未取得anchor run单独模型manifest，身份置信度低于后续run-card/asset-manifest系列。

**缺失与混淆：**

- 本地summary未登记训练seed、完整模型revision、anchor系数/精确损失/初始化方式和完整训练命令，均留缺失。

### full n1：原始simple grader

| 方法/对象 | Step | GSM8K | MATH500 | 记录/来源 |
|---|---:|---:|---:|---|
| anchored_standard | 100 | 35.41 | 12.40 | E067, E075；[S028](#s028) |
| anchored_topk_router | 100 | 37.15 | 13.40 | E069, E077；[S028](#s028) |
| standard | 100 | 35.48 | 13.40 | E071, E079；[S028](#s028) |
| topk_router | 100 | 34.57 | 14.00 | E073, E081；[S028](#s028) |

### full n1：事后primary regrade

| 方法/对象 | Step | GSM8K | MATH500 | 记录/来源 |
|---|---:|---:|---:|---|
| anchored_standard | 100 | 40.03 | 15.20 | E068, E076；[S028](#s028) |
| anchored_topk_router | 100 | 41.93 | 17.80 | E070, E078；[S028](#s028) |
| standard | 100 | 40.71 | 16.80 | E072, E080；[S028](#s028) |
| topk_router | 100 | 39.42 | 17.60 | E074, E082；[S028](#s028) |

**系列证据：** [S027](#s027)、[S028](#s028)。

<a id="early_blocksum50"></a>
## 早期seed7 naive block k1/2/3

**状态：** 完成，k3微弱方向收益且梯度放大，k2负面。**用途：** 历史附录；舍入数值不能作为高精度主表。

**师生身份：** 学生 `Qwen/Qwen3-0.6B`，revision未记录；教师 `Qwen/Qwen3-4B`，revision未记录。
**训练seed：** 7；训练步数 50。协议：`early_cleanroom`。

- 历史run：`/mnt/data/cpfs/Yaleon/opd_cleanroom_20260707/runs/block_opd_effect_20260708_124329`。
- 50steps，2 prompts/step，生成128，prompt1024，LR5e-7，clip1，temperature0.7/top_p0.95，top_k20，stride1/offset0，训练seed7/nonthinking；评测n1/256。
- GSM8K舍入值 token0.4594、k2 0.4541、k3 0.4685；MATH500 0.200/0.194/0.202。只读HTML证据，没有本地原始summary。
- grad norm mean112.68/265.95/473.84，max246/1008/1136；更少loss units不代表更少teacher forward。

**缺失与混淆：**

- train原始 *_regrade_summary.json 本次不可读取；不由样本数把四位小数反算为虚构精确率。
- seed7同名token在pilot50/此run/后续advantage run是独立训练，不能互相替换。

### full n1：文档中的primary舍入值（D级）

| 方法/对象 | Step | GSM8K | MATH500 | 记录/来源 |
|---|---:|---:|---:|---|
| token_opd | 50 | 45.94 | 20.00 | E083；[S029](#s029) |
| block_k2 | 50 | 45.41 | 19.40 | E084；[S029](#s029) |
| block_k3 | 50 | 46.85 | 20.20 | E085；[S029](#s029) |

**系列证据：** [S029](#s029)。

<a id="early_blockadv50"></a>
## 早期seed7 Block3 sum/mean/mixed权重

**状态：** 完成，mean微弱正向；mixed在MATH500负面。**用途：** 历史归一化附录。

**师生身份：** 学生 `Qwen/Qwen3-0.6B`，revision未记录；教师 `Qwen/Qwen3-4B`，revision未记录。
**训练seed：** 7；训练步数 50。协议：`early_cleanroom`。

- 历史run：`/mnt/data/cpfs/Yaleon/opd_cleanroom_20260707/runs/block_advantage_20260708_144223`。
- 同seed7/data/官方非Base师生/50steps，block size3；sum、sum/3、0.5*A_i+0.5*mean(A_B)。训练2prompts/128tokens/LR5e-7/clip1，full eval n1/256。
- GSM8K0.4655/0.4693/0.4708，MATH5000.192/0.198/0.188；这是独立block_advantage run，不能把sum组0.4655混成前一run的0.4685。
- 平均梯度466.74/165.22/83.29；mean降低梯度尺度的观测不是证明跨seed有效或新PPO block ratio的优势。

**缺失与混淆：**

- 原始train JSON未在本次读取范围取得；只有MD/HTML舍入值，grader hash缺失。
- sum/sqrt(k)、lambda0.2/0.8等在文档中是未来计划，未找到完成结果。

### full n1：文档中的primary舍入值（D级）

| 方法/对象 | Step | GSM8K | MATH500 | 记录/来源 |
|---|---:|---:|---:|---|
| block3_sum | 50 | 46.55 | 19.20 | E086；[S002](#s002) |
| block3_mean | 50 | 46.93 | 19.80 | E087；[S002](#s002) |
| block3_mixed_lam05 | 50 | 47.08 | 18.80 | E088；[S002](#s002) |

**系列证据：** [S002](#s002)、[S001](#s001)。

<a id="qwen17_sweep"></a>
## Qwen1.7 Base blocksize sweep k1/3/5/10

**状态：** 完成：k3混合、小幅；k5回退；k10四任务零分。**用途：** 历史尺寸消融，标明跨机器混杂。

**师生身份：** 学生 `Qwen3-1.7B-Base`，revision未记录；教师 `lllyx/Qwen3-4B-Base-GRPO`（本地modelcard身份）；旧run-card revision未记录，缓存commit另见匿名复现补查。
**训练seed：** 21；训练步数 200。协议：`dapo_common`、`qwen_historical17`。

- train A800：Token/Block3/Block5；ml2 A100：Block10。这里只保存/找到Step200四任务n8分数。
- 历史eval未显式固定rollout seeds；external historical grader，不能追认seed21..28或与ml2固定seed结果当完全对照。
- Block5初次Step42 vLLM OOM后新attempt降vLLM预算至0.6；Block10最初numpy/OpenBLAS环境错误后重试。失败现场保留，不记成额外seed。
- k3在MATH500 Avg略升但Pass略降、AIME24 Avg下降；不能写四任务一致提升。sweep k10零分不是全系列唯一collapse证据，另有双机诊断。

**缺失与混淆：**

- 历史k5原始summary没有在本地/ml2查得，使用curated JSON；train只读访问禁用。
- train侧prompt kwargs未在此次逐run核验，历史模板沿用说明与已核验ml2模板区别保留。
- 未找到sweep其余checkpoint已评数值；不能套用后续复跑曲线。

### 旧external，未显式固定eval seed

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| token_opd | 200 | 69.13 / 88.60 | 13.33 / 26.67 | 6.67 / 16.67 | 36.75 / 62.65 | E089；[S075](#s075) |
| block3_mean | 200 | 69.27 / 87.80 | 9.58 / 30.00 | 8.75 / 26.67 | 39.91 / 61.45 | E090；[S076](#s076) |
| block5_mean | 200 | 67.40 / 87.60 | 9.17 / 20.00 | 7.92 / 23.33 | 34.19 / 61.45 | E091；[S030](#s030) |
| block10_mean | 200 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | E092；[S077](#s077) |

**系列证据：** [S030](#s030)、[S031](#s031)、[S029](#s029)。

<a id="block10_collapse"></a>
## Block10双机独立诊断复跑

**状态：** 两机完成但均发生科学失败；Step200全零。**用途：** 必须保留的负面/稳定性附录。

**师生身份：** 学生 `Qwen3-1.7B-Base`，revision未记录；教师 `lllyx/Qwen3-4B-Base-GRPO`（本地modelcard身份）；旧run-card revision未记录，缓存commit另见匿名复现补查。
**训练seed：** 21；训练步数 200。协议：`dapo_common`、`qwen_historical17`。

- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260710v5_block10_collapse_diag_seed21_ml2/block10_mean`；runtime `af05c071e422fca02c5bbfba3f84277208b83b1e`；block10_mean，k=undefined/undefined；run card [S198](#s198)，训练验收 [S199](#s199)。配置存在不代表开训。
- 实验20260710v5，runtime af05c071e422fca02c5bbfba3f84277208b83b1e；A800和A100相同seed21，不是两个训练seed。
- 固定k10 mean advantage + joint PPO ratio + block reduction联合改变；不是只广播mean advantage。两机共同诊断步prompt hashes匹配。
- 两机评测均为builtin VERL，seeds21..28，与sweep旧external分数不合并；AMC23 builtin零分不能单独证明能力全无。
- 两机Step200 MATH500/AIME24/AIME25/AMC23全零且输出严重退化，数值有限、无OOM，训练exit0；工程验收通过不代表方法成功。
- support drift/ratio放大先后不跨机一致；不支持唯一credit-leakage根因或tail-to-front传播，不能从相关性作因果结论。

**缺失与混淆：**

- checkpoint40/60/80保留但未找到完整benchmark分数；只报告50/100/200。
- 没有同runtime同机新Token完整配对或多seed因果消融；两机相同seed不能估训练方差。

### builtin VERL（敏感性视图，不能替换external）

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| block10_mean:train-A800 | 50 | 45.20 / 79.20 | 2.50 / 13.33 | 2.92 / 10.00 | 0.00 / 0.00 | E093；[S032](#s032) |
| block10_mean:train-A800 | 100 | 2.70 / 16.60 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | E094；[S032](#s032) |
| block10_mean:train-A800 | 200 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | E095；[S032](#s032) |
| block10_mean:ml2-A100 | 50 | 37.75 / 78.60 | 4.17 / 20.00 | 2.08 / 13.33 | 0.00 / 0.00 | E096；[S078](#s078) |
| block10_mean:ml2-A100 | 100 | 0.97 / 7.20 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | E097；[S079](#s079) |
| block10_mean:ml2-A100 | 200 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | E098；[S080](#s080) |

**系列证据：** [S032](#s032)、[S033](#s033)。

<a id="qwen17_ml2"></a>
## Qwen1.7 Base 同机Token/Block3严格配对

**状态：** 完成，Step200四任务单seed正向；Step100存在负面。**用途：** 历史受控配对可入历史主表，不能外推新completion协议。

**师生身份：** 学生 `Qwen3-1.7B-Base`，revision未记录；教师 `lllyx/Qwen3-4B-Base-GRPO`（本地modelcard身份）；旧run-card revision未记录，缓存commit另见匿名复现补查。
**训练seed：** 21；训练步数 200。协议：`dapo_common`、`qwen_historical17`。

- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd`；runtime `9b3b8b76bcdf02d0a4cfe4720cecb24bc7203977`；token_opd，k=undefined/undefined；run card [S003](#s003)，训练验收 [S195](#s195)。配置存在不代表开训。
- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean`；runtime `9b3b8b76bcdf02d0a4cfe4720cecb24bc7203977`；block3_mean，k=undefined/undefined；run card [S196](#s196)，训练验收 [S197](#s197)。配置存在不代表开训。
- ml2 4xA100，runtime9b3b8b76bcdf02d0a4cfe4720cecb24bc7203977；Token与Block3各200steps，50/100/200完整state/eval；micro1/4/1、vLLM0.6。
- 两组41个诊断prompt hashes匹配；同机baseline晚于单Block3复现完成，旧HTML的'没有同机Token'已过时。
- 主来源选择最终historical_external_grader_audited/summary.json；builtin独立保留。最终本地comparisons两种grader JSON均匹配finalization_hashes，且external所有分数与ml2原始final摘要一致。
- Step200 external四任务Avg/Pass均比Token高；Step100 MATH500 Avg下降，AIME24 Avg/Pass下降，不能隐藏中间checkpoint。
- 原历史train分数高于ml2不证明硬件因果：eval seeds、reference micro4->1、vLLM0.7->0.6等一起变化。
- 一个训练seed；原有macro bootstrap只估固定checkpoint题目不确定性，不作为各任务CI。

**缺失与混淆：**

- 学生Hub revision未登记；不得借用Qwen/Qwen3-1.7B的另一份manifest。教师repo由本地modelcard补齐，缓存commit证据独立记录，旧run-card未追补。
- 原始训练全量token轨迹没有保留；不能用9月重生成补称原训练raw。

### builtin VERL（敏感性视图，不能替换external）

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| token_opd | 50 | 37.05 / 78.00 | 3.75 / 16.67 | 2.50 / 13.33 | 0.00 / 0.00 | E099；[S082](#s082) |
| token_opd | 100 | 41.80 / 80.20 | 6.25 / 26.67 | 2.08 / 6.67 | 0.00 / 0.00 | E101；[S085](#s085) |
| token_opd | 200 | 44.45 / 80.00 | 5.42 / 20.00 | 2.50 / 10.00 | 0.00 / 0.00 | E103；[S088](#s088) |
| block3_mean | 50 | 50.25 / 81.00 | 5.42 / 20.00 | 3.75 / 16.67 | 0.00 / 0.00 | E105；[S091](#s091) |
| block3_mean | 100 | 40.80 / 80.40 | 5.00 / 20.00 | 3.33 / 13.33 | 0.00 / 0.00 | E107；[S094](#s094) |
| block3_mean | 200 | 52.83 / 82.20 | 8.75 / 26.67 | 5.42 / 16.67 | 0.00 / 0.00 | E109；[S097](#s097) |

### 最终historical external audited

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| token_opd | 50 | 38.07 / 80.20 | 3.75 / 16.67 | 2.50 / 13.33 | 13.70 / 45.78 | E100；[S083](#s083) |
| token_opd | 100 | 43.10 / 82.20 | 6.25 / 26.67 | 2.08 / 6.67 | 18.67 / 57.83 | E102；[S086](#s086) |
| token_opd | 200 | 45.85 / 82.00 | 5.42 / 20.00 | 2.50 / 10.00 | 21.84 / 54.22 | E104；[S089](#s089) |
| block3_mean | 50 | 52.10 / 83.60 | 5.42 / 20.00 | 3.75 / 16.67 | 22.14 / 49.40 | E106；[S092](#s092) |
| block3_mean | 100 | 42.27 / 82.20 | 5.00 / 20.00 | 3.33 / 13.33 | 23.64 / 61.45 | E108；[S095](#s095) |
| block3_mean | 200 | 54.75 / 84.80 | 8.75 / 26.67 | 5.42 / 16.67 | 27.26 / 61.45 | E110；[S098](#s098) |

### 同step配对差值

方向：`block3_mean - token_opd`；下面为 **Avg@8差值 / Pass@8差值（百分点）**。仅同系列、同grader和同prompt；不是相对百分比，不附虚构的任务CI。

| Step | MATH500 | AIME24 | AIME25 | AMC23 |
|---:|---:|---:|---:|---:|
| 50 | 14.03 / 3.40 | 1.67 / 3.33 | 1.25 / 3.33 | 8.43 / 3.61 |
| 100 | -0.82 / 0.00 | -1.25 / -6.67 | 1.25 / 6.67 | 4.97 / 3.61 |
| 200 | 8.90 / 2.80 | 3.33 / 6.67 | 2.92 / 6.67 | 5.42 / 7.23 |

**系列证据：** [S034](#s034)、[S035](#s035)、[S036](#s036)、[S031](#s031)。

<a id="qwen17_reeval"></a>
## Qwen1.7历史六权重新生成评测

**状态：** 六项完成；非新训练，非新增seed。**用途：** 重生成敏感性/可追溯附录。

**师生身份：** 学生 `Qwen3-1.7B-Base`，revision未记录；教师 `lllyx/Qwen3-4B-Base-GRPO`（本地modelcard身份）；旧run-card revision未记录，缓存commit另见匿名复现补查。
**训练seed：** 21；本次不新增训练。协议：`dapo_common`、`qwen_historical17`。

- 来源权重仍为7月ml2 Token/Block3 Step50/100/200；20260918v2新runtime94be7ea仅重新评测并保存lossless raw。
- 保持历史external grader、四任务n8/seeds21..28与输入prompt IDs；新生成分数与7月略不同，不能把9月数值覆盖7月原始结果。
- 该大队列后来Llama probe失败，但qwen17/acceptance与6个per-eval acceptance通过；父队列failed不能倒推六个评测失败。
- 原始7月Step200 MATH500 Token0.4585/Block3 0.5475；9月再生成0.45925/0.5415。保留两个view，不平均、不择优。

**缺失与混淆：**

- 没有新增训练seed/新模型对证据；固定采样seed不保证跨runtime逐位重复。

### historical external

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| token_opd | 50 | 38.30 / 81.80 | 2.50 / 16.67 | 3.75 / 16.67 | 14.61 / 51.81 | E135；[S119](#s119) |
| token_opd | 100 | 43.35 / 83.00 | 6.25 / 20.00 | 2.08 / 6.67 | 18.52 / 55.42 | E136；[S122](#s122) |
| token_opd | 200 | 45.92 / 83.00 | 6.67 / 16.67 | 3.75 / 10.00 | 23.34 / 60.24 | E137；[S125](#s125) |
| block3_mean | 50 | 52.05 / 84.60 | 5.00 / 16.67 | 3.33 / 16.67 | 23.80 / 51.81 | E138；[S128](#s128) |
| block3_mean | 100 | 42.65 / 83.00 | 5.00 / 20.00 | 2.50 / 13.33 | 23.49 / 56.63 | E139；[S131](#s131) |
| block3_mean | 200 | 54.15 / 84.00 | 7.08 / 26.67 | 5.83 / 16.67 | 29.22 / 65.06 | E140；[S134](#s134) |

### 同step配对差值

方向：`block3_mean - token_opd`；下面为 **Avg@8差值 / Pass@8差值（百分点）**。仅同系列、同grader和同prompt；不是相对百分比，不附虚构的任务CI。

| Step | MATH500 | AIME24 | AIME25 | AMC23 |
|---:|---:|---:|---:|---:|
| 50 | 13.75 / 2.80 | 2.50 / 0.00 | -0.42 / 0.00 | 9.19 / 0.00 |
| 100 | -0.70 / 0.00 | -1.25 / 0.00 | 0.42 / 6.67 | 4.97 / 1.20 |
| 200 | 8.22 / 1.00 | 0.42 / 10.00 | 2.08 / 6.67 | 5.87 / 4.82 |

**系列证据：** [S037](#s037)、[S038](#s038)、[S039](#s039)。

<a id="deepseek_justrl"></a>
## DeepSeek-R1-Distill/JustRL配对

**状态：** 完成；未达到预定跨师生方向复现。**用途：** 必须保留的跨模型负面对照。

**师生身份：** 学生 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`，revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`；教师 `hbx/JustRL-DeepSeek-1.5B`，revision `0637e4096c789c67f9eecbe8355e0bdeddede1c2`。
**训练seed：** 21；训练步数 200。协议：`dapo_common`。

- 两组均在train 4xA800，runtime87274bfb3d1956a385fd0f43591319cb723577d2；本次未连接train。师生1.5B同架构、同tokenizer的既有核验。
- 每臂200steps，41诊断prompt batches匹配，50/100/200完整n8四任务。外部与builtin grader各自保留；finalization哈希逐字节匹配本地四份配对JSON。
- Step200 MATH500 Avg微升但Pass下降，AIME24 Avg升而Pass下降；AIME25升，AMC23 Pass不变。不能仅挑Avg或终点有利项。
- Step50 AIME24和AMC23退步，Step100 AIME25明显退步；历史预定复现门槛未满足。
- 两臂无记录到的数值崩塌；teacher forward未减少，耗时18950.954/18767.403秒约-0.97%不构成实质加速。

**缺失与混淆：**

- 本次没有train原始eval_summary/raw逐题重验；B级最终化JSON证据不是新的raw审计。
- 训练rendered prompt/kwargs/原生停止配置缺逐样本原始记录；不把计划的eval thinking=false写成已证实训练全程关闭thinking。
- 一个训练seed；未找到原始学生/教师匹配完整benchmark分数。

### historical external

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| token_opd | 50 | 83.78 / 94.80 | 38.75 / 73.33 | 27.08 / 50.00 | 71.69 / 90.36 | E111；[S006](#s006) |
| block3_mean | 50 | 84.15 / 94.60 | 32.50 / 66.67 | 27.50 / 43.33 | 70.93 / 86.75 | E112；[S006](#s006) |
| token_opd | 100 | 84.63 / 94.20 | 39.17 / 70.00 | 34.58 / 46.67 | 73.19 / 89.16 | E113；[S006](#s006) |
| block3_mean | 100 | 84.45 / 94.80 | 39.17 / 73.33 | 25.42 / 36.67 | 72.74 / 90.36 | E114；[S006](#s006) |
| token_opd | 200 | 84.70 / 94.60 | 39.58 / 70.00 | 29.17 / 46.67 | 73.34 / 90.36 | E115；[S006](#s006) |
| block3_mean | 200 | 84.75 / 94.00 | 43.75 / 66.67 | 30.83 / 50.00 | 73.80 / 90.36 | E116；[S006](#s006) |

### builtin VERL（敏感性视图，不能替换external）

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| token_opd | 50 | 81.47 / 91.60 | 38.75 / 73.33 | 27.08 / 50.00 | 0.00 / 0.00 | E117；[S040](#s040) |
| block3_mean | 50 | 81.80 / 91.60 | 32.50 / 66.67 | 27.50 / 43.33 | 0.00 / 0.00 | E118；[S040](#s040) |
| token_opd | 100 | 82.13 / 91.00 | 39.17 / 70.00 | 34.58 / 46.67 | 0.00 / 0.00 | E119；[S040](#s040) |
| block3_mean | 100 | 82.23 / 91.80 | 39.17 / 73.33 | 25.42 / 36.67 | 0.00 / 0.00 | E120；[S040](#s040) |
| token_opd | 200 | 82.25 / 91.60 | 39.58 / 70.00 | 29.17 / 46.67 | 0.00 / 0.00 | E121；[S040](#s040) |
| block3_mean | 200 | 82.25 / 90.80 | 43.75 / 66.67 | 30.83 / 50.00 | 0.00 / 0.00 | E122；[S040](#s040) |

### 同step配对差值

方向：`block3_mean - token_opd`；下面为 **Avg@8差值 / Pass@8差值（百分点）**。仅同系列、同grader和同prompt；不是相对百分比，不附虚构的任务CI。

| Step | MATH500 | AIME24 | AIME25 | AMC23 |
|---:|---:|---:|---:|---:|
| 50 | 0.38 / -0.20 | -6.25 / -6.67 | 0.42 / -6.67 | -0.75 / -3.61 |
| 100 | -0.17 / 0.60 | 0.00 / 3.33 | -9.17 / -10.00 | -0.45 / 1.20 |
| 200 | 0.05 / -0.60 | 4.17 / -3.33 | 1.67 / 3.33 | 0.45 / 0.00 |

**系列证据：** [S006](#s006)、[S040](#s040)、[S041](#s041)、[S005](#s005)、[S042](#s042)。

<a id="window_random_sliding"></a>
## Qwen1.7 random3/sliding3

**状态：** 两臂及50/100/200评测完成；sliding相对random后期正向，但两者退化。**用途：** 窗口方法对照附录；不能冒充fixed3对照。

**师生身份：** 学生 `Qwen3-1.7B-Base`，revision未记录；教师 `lllyx/Qwen3-4B-Base-GRPO`（本地modelcard身份）；旧run-card revision未记录，缓存commit另见匿名复现补查。
**训练seed：** 21；训练步数 200。协议：`dapo_common`、`qwen_historical17`。

- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3`；runtime `fbad852a638de18e20d571a061b2be0437942a38`；random3，k=3/mean；run card [S200](#s200)，训练验收 [S201](#s201)。配置存在不代表开训。
- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3`；runtime `fbad852a638de18e20d571a061b2be0437942a38`；sliding3，k=3/mean；run card [S202](#s202)，训练验收 [S203](#s203)。配置存在不代表开训。
- 20260911v2r1，runtimefbad852a638de18e20d571a061b2be0437942a38；seed21；random offset RNG PCG64 seed910021，三offset0/1/2，逐step全rank共享并保存恢复。
- random3随机选一个offset；sliding3对三个offset各自完整phase loss求平均，不是先平均advantage再求一次loss。phase loss按有效window长度加权；保留首尾partial windows，一个optimizer update/step。
- 更正此前“归一化不同”的过强概括：对相同连续有效块、相同每块surrogate C_b，令N=sum_b n_b>0，则历史weighted reduction为sum_b[(n_b/k) C_b]/(N/k+eta)=sum_b[n_b C_b]/(N+k*eta)。eta=0时与按window counts加权、分母为N的写法代数相同；不能仅凭token-normalized名称断言额外主要尺度混淆。
- 完整phase-loss平均、mask空洞/首尾partial-window构造及数值稳定项需逐项区分，具体实现差异及影响待数学侧核查，不将这些边界情形泛化为整体normalization差异。没有新Token/fixed3同runtime训练，旧fixed3仍仅为contextual对照，不能据此建立纯边界处理的因果收益。
- 200步有序prompt顺序及41诊断hash匹配；原始队列在JSONL/审计环节失败，独立analysis恢复后complete，训练runtime未热改。
- external Step50 sliding的MATH/AIME/AMC Avg均较random低；Step100和200四任务Avg较高，但sliding MATH500 Avg从0.52075降至0.3475；并不稳定消除退化。
- 计划中seeds22/23、64题probe和新Base/Token/fixed eval未执行；不计入结果。

**缺失与混淆：**

- 无fresh fixed3/Token controls；训练组内8条的独立性未通过全量raw重新验证。

### historical external

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| random3 | 50 | 45.23 / 81.60 | 5.00 / 20.00 | 2.92 / 10.00 | 18.22 / 48.19 | E123；[S100](#s100) |
| random3 | 100 | 36.70 / 82.40 | 2.08 / 13.33 | 1.67 / 10.00 | 14.91 / 48.19 | E125；[S103](#s103) |
| random3 | 200 | 21.18 / 73.60 | 3.33 / 16.67 | 0.00 / 0.00 | 12.20 / 46.99 | E127；[S106](#s106) |
| sliding3 | 50 | 41.33 / 81.80 | 3.75 / 23.33 | 1.67 / 10.00 | 15.66 / 48.19 | E129；[S109](#s109) |
| sliding3 | 100 | 52.08 / 85.20 | 4.58 / 20.00 | 2.92 / 10.00 | 24.85 / 57.83 | E131；[S112](#s112) |
| sliding3 | 200 | 34.75 / 81.00 | 5.00 / 20.00 | 2.08 / 10.00 | 15.81 / 48.19 | E133；[S115](#s115) |

### builtin VERL（敏感性视图，不能替换external）

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| random3 | 50 | 43.43 / 79.00 | 5.00 / 20.00 | 2.92 / 10.00 | 0.00 / 0.00 | E124；[S101](#s101) |
| random3 | 100 | 35.27 / 79.60 | 2.08 / 13.33 | 1.67 / 10.00 | 0.00 / 0.00 | E126；[S104](#s104) |
| random3 | 200 | 20.50 / 71.60 | 3.33 / 16.67 | 0.00 / 0.00 | 0.00 / 0.00 | E128；[S107](#s107) |
| sliding3 | 50 | 40.02 / 79.20 | 3.75 / 23.33 | 1.67 / 10.00 | 0.00 / 0.00 | E130；[S110](#s110) |
| sliding3 | 100 | 50.42 / 82.80 | 4.58 / 20.00 | 2.92 / 10.00 | 0.00 / 0.00 | E132；[S113](#s113) |
| sliding3 | 200 | 33.70 / 78.00 | 5.00 / 20.00 | 2.08 / 10.00 | 0.00 / 0.00 | E134；[S116](#s116) |

### 同step配对差值

方向：`sliding3 - random3`；下面为 **Avg@8差值 / Pass@8差值（百分点）**。仅同系列、同grader和同prompt；不是相对百分比，不附虚构的任务CI。

| Step | MATH500 | AIME24 | AIME25 | AMC23 |
|---:|---:|---:|---:|---:|
| 50 | -3.90 / 0.20 | -1.25 / 3.33 | -1.25 / 0.00 | -2.56 / 0.00 |
| 100 | 15.38 / 2.80 | 2.50 / 6.67 | 1.25 / 0.00 | 9.94 / 9.64 |
| 200 | 13.57 / 7.40 | 1.67 / 3.33 | 2.08 / 10.00 | 3.61 / 1.20 |

**系列证据：** [S043](#s043)、[S044](#s044)、[S045](#s045)、[S046](#s046)。

<a id="qwen06_old_prompt"></a>
## Qwen0.6 Base旧thinking-unspecified提示

**状态：** Token训练完成；Step50评测用户中止；Block3未启动。**用途：** 历史中止记录；无完整配对分数。

**师生身份：** 学生 `Qwen/Qwen3-0.6B-Base`，revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`；教师 `lllyx/Qwen3-4B-Base-GRPO`（本地modelcard身份）；旧run-card revision未记录，缓存commit另见匿名复现补查。
**训练seed：** 21；训练步数 200。协议：`dapo_common`、`qwen_historical17`。

- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v1_qwen06_pair_seed21_ml2/token_opd`；runtime `ec0a7a950540d7f4753c08b18bc26c544f6a7cde`；token_opd，k=1/sum；run card [S204](#s204)，训练验收 [S205](#s205)。配置存在不代表开训。
- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v1_qwen06_pair_seed21_ml2/block3_mean`；runtime `ec0a7a950540d7f4753c08b18bc26c544f6a7cde`；block3_mean，k=3/mean；run card [S206](#s206)，未查得正式训练验收。配置存在不代表开训。
- 20260913v1，runtimeec0a7a950540d7f4753c08b18bc26c544f6a7cde；Token200steps、50/100/200 checkpoints；非早期非Base0.6B。
- 用户9月13日14:04主动停止评测，error143/eval -15，不是自发训练失败；Block3 run card存在不证明开训。
- 训练全程长度触顶率39.125%，旧1.7 Token13.625%，但模型/代码身份不同且不是因果参数量消融。
- 原训练raw没有保存，后续192条推理诊断不是补回历史轨迹。

**缺失与混淆：**

- Step50/100/200四任务完整已验收benchmark分数缺失；本次未找到旧Step50全套summary，不能用后续非thinking Token分数填补。
- Block3未训练/未评；旧Token原始training trajectories不可恢复。

### 原计划VERL评测，未完整完成

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| token_opd | 50 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | E166；[S048](#s048) |
| token_opd | 100 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | E167；[S048](#s048) |
| token_opd | 200 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | E168；[S048](#s048) |

**系列证据：** [S014](#s014)、[S047](#s047)、[S048](#s048)。

<a id="qwen06_nonthinking"></a>
## Qwen0.6 Base显式非thinking配对

**状态：** Token/Block3训练与全评完成；Token提升、Block3严重负面。**用途：** 必须保留的同协议小模型负面对照。

**师生身份：** 学生 `Qwen/Qwen3-0.6B-Base`，revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`；教师 `lllyx/Qwen3-4B-Base-GRPO`（本地modelcard身份）；旧run-card revision未记录，缓存commit另见匿名复现补查。
**训练seed：** 21；训练步数 200。协议：`dapo_common`、`qwen_nonthinking`。

- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v4_qwen06_nonthinking_token_seed21_ml2/token_opd`；runtime `0ce73aa42d7f734b9d34c379f474458bb6d48771`；token_opd，k=1/sum；run card [S207](#s207)，训练验收 [S208](#s208)。配置存在不代表开训。
- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/block3_mean`；runtime `0ce73aa42d7f734b9d34c379f474458bb6d48771`；block3_mean，k=3/mean；run card [S209](#s209)，训练验收 [S210](#s210)。配置存在不代表开训。
- Token20260913v4；Token/Base/teacher评测及Block3训练20260917v1；Block3评测20260917v2。runtime固定0ce73aa。200步首尾原始输入/顺序匹配。
- v2 GPU门禁因fixture AttributeError在generation前失败；v3因CUDA初始化后fork失败；v4用spawn通过。不是三个成功训练seed。
- 五个Token/Base/teacher视图 + 三个Block3 checkpoints均per-eval acceptance通过。旧文档说Block3未评是9月17日快照，9月18日00:53UTC队列已complete。
- Block3 Step200 MATH500 Avg0.00575/Pass0.044，AMC23 0.004518072289156626/0.024096385542168676，两AIME全零；相同prompt Token对应MATH0.42875/0.756，AMC0.1897590361445783/0.46987951807228917。
- Block3 6400名义轨迹有4080真实length stop；同题8条重复，不能称6400独立rollout。Token也存在空白重复，不能宣称全部健康。
- 与旧0.6B对比同时涉及prompt/EOS/mask修正与版本变化；新组内部Token/Block3可控，不把跨组改善只归因关闭thinking。

**缺失与混淆：**

- 没有训练seed重复；输出退化因果机制未隔离。
- 本次读取既有acceptance，不重新下载/判分15432条Block3原始输出。

### historical external

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| student_base | 0 | 10.45 / 45.20 | 0.42 / 3.33 | 0.00 / 0.00 | 3.46 / 24.10 | E141；[S137](#s137) |
| teacher | 0 | 83.43 / 93.40 | 23.33 / 43.33 | 18.75 / 26.67 | 57.68 / 77.11 | E142；[S140](#s140) |
| token_opd | 50 | 25.70 / 65.40 | 1.25 / 6.67 | 0.42 / 3.33 | 10.09 / 42.17 | E143；[S143](#s143) |
| token_opd | 100 | 39.60 / 72.20 | 2.50 / 13.33 | 1.25 / 10.00 | 15.36 / 38.55 | E144；[S146](#s146) |
| token_opd | 200 | 42.88 / 75.60 | 2.50 / 10.00 | 0.83 / 6.67 | 18.98 / 46.99 | E145；[S149](#s149) |
| block3_mean | 50 | 16.20 / 56.60 | 0.00 / 0.00 | 0.42 / 3.33 | 5.87 / 26.51 | E146；[S152](#s152) |
| block3_mean | 100 | 3.48 / 23.60 | 0.00 / 0.00 | 0.00 / 0.00 | 1.05 / 6.02 | E147；[S155](#s155) |
| block3_mean | 200 | 0.57 / 4.40 | 0.00 / 0.00 | 0.00 / 0.00 | 0.45 / 2.41 | E148；[S158](#s158) |

### 同step配对差值

方向：`block3_mean - token_opd`；下面为 **Avg@8差值 / Pass@8差值（百分点）**。仅同系列、同grader和同prompt；不是相对百分比，不附虚构的任务CI。

| Step | MATH500 | AIME24 | AIME25 | AMC23 |
|---:|---:|---:|---:|---:|
| 50 | -9.50 / -8.80 | -1.25 / -6.67 | 0.00 / 0.00 | -4.22 / -15.66 |
| 100 | -36.13 / -48.60 | -2.50 / -13.33 | -1.25 / -10.00 | -14.31 / -32.53 |
| 200 | -42.30 / -71.20 | -2.50 / -10.00 | -0.83 / -6.67 | -18.52 / -44.58 |

**系列证据：** [S049](#s049)、[S050](#s050)、[S051](#s051)。

<a id="llama_native_stopped"></a>
## Llama3.2原生非thinking提示尝试

**状态：** Base评测完成；Token在Step146被用户停止；Block3未启动。**用途：** 历史中止记录，不构成方法对照。

**师生身份：** 学生 `LLM-Research/Llama-3.2-1B-Instruct`，revision `d3e551343d4d81508a0d226656b826c217e463cd`；教师 `LLM-Research/Llama-3.2-3B-Instruct`，revision `4e7231b81c151c73632184994ac9a0149fcb22fd`。
**训练seed：** 21；训练步数 146。协议：`dapo_common`、`llama_native`。

- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v1_llama32_1b_3b_nonthinking_seed21_ml2/token_opd`；runtime `f2d148e74c06283617a878b1a03fabdd5ef390da`；token_opd，k=1/sum；run card [S211](#s211)，未查得正式训练验收。配置存在不代表开训。
- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v1_llama32_1b_3b_nonthinking_seed21_ml2/block3_mean`；runtime `f2d148e74c06283617a878b1a03fabdd5ef390da`；block3_mean，k=3/mean；run card [S212](#s212)，未查得正式训练验收。配置存在不代表开训。
- ModelScope原始1B-Instruct学生/3B-Instruct教师，不是Llama Base或Qwen ChatML。20260918v1，runtimef2d148e74c06283617a878b1a03fabdd5ef390da。
- Base完整四任务n8已验收；Token保存50/100完整state与146步raw，但没有Step200；后续方法checkpoint eval未开展。
- 用户12:01:41北京要求更换为历史训练提示，从原始学生新开；error143不是自发训练崩塌。

**缺失与混淆：**

- Token50/100/200 benchmark分数未找到；200 checkpoint不存在。
- 没有Block3训练/评测，不能用于原生vs历史prompt的训练效果消融。

### historical external

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| student_base | 0 | 13.80 / 45.20 | 0.42 / 3.33 | 0.00 / 0.00 | 4.07 / 21.69 | E149；[S161](#s161) |
| token_opd | 50 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | E169；[S052](#s052) |
| token_opd | 100 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | E170；[S052](#s052) |
| token_opd | 200 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | E171；[S052](#s052) |

**系列证据：** [S037](#s037)、[S052](#s052)。

<a id="llama_historical"></a>
## Llama3.2 historical17训推差异配对

**状态：** 两组200步及全部评测完成；Block3明显负面。**用途：** 必须保留的跨模型负面对照。

**师生身份：** 学生 `LLM-Research/Llama-3.2-1B-Instruct`，revision `d3e551343d4d81508a0d226656b826c217e463cd`；教师 `LLM-Research/Llama-3.2-3B-Instruct`，revision `4e7231b81c151c73632184994ac9a0149fcb22fd`。
**训练seed：** 21；训练步数 200。协议：`dapo_common`、`llama_historical`。

- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/token_opd`；runtime `94be7ea1d659309256c8356681925bb9710895c4`；token_opd，k=1/sum；run card [S213](#s213)，训练验收 [S214](#s214)。配置存在不代表开训。
- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/block3_mean`；runtime `94be7ea1d659309256c8356681925bb9710895c4`；block3_mean，k=3/mean；run card [S215](#s215)，训练验收 [S216](#s216)。配置存在不代表开训。
- 20260918v2完成Qwen六权重重评和Llama初始评测后，Token probe因Ray AF_UNIX路径>107字节而失败，尚未更新。
- v3 bootstrap在manifest/Ray前因NFS只读fd exclusive flock失败；v4短cache恢复，controllerb00ad933，training/eval仍94be7ea。两臂从原始学生重开，不续旧Step146。
- v4 9月19日13:42北京complete，复用v2的已验收初始学生评测，不复用旧v1的另一批Base采样；这两批Base分数也都保留。
- Block3 Step200 MATH500 Avg0.063/Pass0.284，Token0.1555/0.452；AIME24降至0、AIME25两组0，AMC23明显下降。
- Step100 Block3在AIME24/25有局部较高值，不能只报终点或称全程所有任务都零；所有50/100/200完整分数保留。
- 200步实际prompt/order/sampling/protocol配对，首rollouts相等。Block3 mean同时含joint ratio与block reduction，不是纯mean广播；等advantage toy的logprob梯度3倍不等于Adam更新3倍。
- 师生熵升高/低熵重复都可伴随退化；正式Block3从至少Step28有坏输出，最大grad Step75不是所有退化的起点。

**缺失与混淆：**

- 无多seed；没有同预算完成的原生prompt Token/Block3两臂，所以不能独立证明historical训练提示是主因。
- 教师完整四benchmark分数未找到；16题prompt probe不能替代。

### historical external

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| student_base | 0 | 14.00 / 46.40 | 0.42 / 3.33 | 0.00 / 0.00 | 5.27 / 28.92 | E150；[S164](#s164) |
| token_opd | 50 | 14.15 / 44.80 | 0.00 / 0.00 | 0.00 / 0.00 | 6.78 / 32.53 | E151；[S167](#s167) |
| token_opd | 100 | 14.90 / 45.80 | 0.42 / 3.33 | 0.00 / 0.00 | 6.48 / 31.33 | E152；[S170](#s170) |
| token_opd | 200 | 15.55 / 45.20 | 0.42 / 3.33 | 0.00 / 0.00 | 6.02 / 28.92 | E153；[S173](#s173) |
| block3_mean | 50 | 6.13 / 27.20 | 0.00 / 0.00 | 0.00 / 0.00 | 1.51 / 8.43 | E154；[S176](#s176) |
| block3_mean | 100 | 11.15 / 38.60 | 1.25 / 6.67 | 0.42 / 3.33 | 4.52 / 22.89 | E155；[S179](#s179) |
| block3_mean | 200 | 6.30 / 28.40 | 0.00 / 0.00 | 0.00 / 0.00 | 1.51 / 9.64 | E156；[S182](#s182) |

### 同step配对差值

方向：`block3_mean - token_opd`；下面为 **Avg@8差值 / Pass@8差值（百分点）**。仅同系列、同grader和同prompt；不是相对百分比，不附虚构的任务CI。

| Step | MATH500 | AIME24 | AIME25 | AMC23 |
|---:|---:|---:|---:|---:|
| 50 | -8.02 / -17.60 | 0.00 / 0.00 | 0.00 / 0.00 | -5.27 / -24.10 |
| 100 | -3.75 / -7.20 | 0.83 / 3.33 | 0.42 / 3.33 | -1.96 / -8.43 |
| 200 | -9.25 / -16.80 | -0.42 / -3.33 | 0.00 / 0.00 | -4.52 / -19.28 |

**系列证据：** [S053](#s053)、[S054](#s054)、[S055](#s055)、[S056](#s056)。

<a id="qwen4_chatml_stopped"></a>
## Qwen4旧ChatML Block3-first失败/中止

**状态：** 正式Step4用户中止；无正式checkpoint/eval。**用途：** 协议失败与负面诊断附录。

**师生身份：** 学生 `Qwen/Qwen3-4B-Base`，revision `bbd6fc8d23e8788d987b7b970cbb7bd31c826e38`；教师 `lllyx/Qwen3-4B-Base-GRPO`（本地modelcard身份）；旧run-card revision未记录，缓存commit另见匿名复现补查。
**训练seed：** 21；训练步数 4。协议：`dapo_common`、`qwen_historical17`。

- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260919v1_qwen4_blockfirst_seed21_ml2/block3_mean`；runtime `57ae7dfec10e206dd441a4c70571308c09f64a03`；block3_mean，k=3/mean；run card [S217](#s217)，未查得正式训练验收。配置存在不代表开训。
- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260919v1_qwen4_blockfirst_seed21_ml2/token_opd`；runtime `57ae7dfec10e206dd441a4c70571308c09f64a03`；token_opd，k=1/sum；run card [S218](#s218)，未查得正式训练验收。配置存在不代表开训。
- Qwen4只是本项目4B学生简称，实际模型是Qwen3-4B-Base；不是Qwen4代模型。20260919v1 runtime57ae7dfec10e206dd441a4c70571308c09f64a03。
- 9月19日21:32用户停止/error143；完成4次更新、4批实际raw，未到Step50无正式checkpoint；仅probe1/2 state保留。
- 第一次更新前就有16/32名义长度截断，仅4题各重复8次。独立原始权重vLLM/HF诊断支持Base/ChatML失配为诱因，不可全部归因Block3更新。
- 删除think指令不等于去掉ChatML；只删三个marker与plain+Solution是独立干预，不能与新completion正式分数拼表。
- 固定每请求seed放大组内重复但不是唯一原因；修新prompt同时修request-seed，旧新不是单一变量消融。

**缺失与混淆：**

- 没有正式50/100/150/200 checkpoint及所有benchmark分数；不得填0或用probe得分替代。
- 新completion依然可能有重复；诊断不是长程安全性证明。

**系列证据：** [S057](#s057)、[S058](#s058)、[S059](#s059)。

<a id="qwen4_completion"></a>
## Qwen4新completion Block3与新Token

**状态：** Block3四checkpoint全评完成；Token200步训练已完，benchmark等待。**用途：** 候选新协议主表，匹配Base/Token评测未齐前不可写增益。

**师生身份：** 学生 `Qwen/Qwen3-4B-Base`，revision `bbd6fc8d23e8788d987b7b970cbb7bd31c826e38`；教师 `lllyx/Qwen3-4B-Base-GRPO`（本地modelcard身份）；旧run-card revision未记录，缓存commit另见匿名复现补查。
**训练seed：** 21；训练步数 200。协议：`dapo_common`、`qwen_completion`。

- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2/block3_mean`；runtime `0f9161f02f08287fb07f0375ad0a6bda81133ff0`；block3_mean，k=3/mean；run card [S219](#s219)，训练验收 [S220](#s220)。配置存在不代表开训。
- Run `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_token_seed21_ml2/token_opd`；runtime `0f9161f02f08287fb07f0375ad0a6bda81133ff0`；token_opd，k=1/sum；run card [S221](#s221)，训练验收 [S222](#s222)。配置存在不代表开训。
- 两个原始模型16题/192输出诊断、Block3与Token各Step1保存恢复Step2 gate已通过，不计正式benchmark。
- completion正式Block3 v2在第一次更新前Triton NFS TemporaryDirectory cleanup OSError39失败，保存32条首rollout；不是GPU OOM或算法发散。v3只更换临时cache/失败清理，同runtime重开原始学生。
- Block3正式20260920v3 200steps，四checkpoint50/100/150/200；eval队列20260920v1于9月20日14:43:49北京完成四项全量验收，原本local progress只记Step200已过时。
- Token20260920v1于9月21日00:29:02北京训练complete；acceptance.passed，50/100/150/200 checkpoint，paired_rollouts_verified=true；full_eval_started=false是该训练控制器状态，不能据此判断父线程随后新评测队列状态。
- 本审计不启动/等待父线程新Token评测，不填任何新Token benchmark值。Block3只有绝对分数，不是相对新Token/Base的增益。
- Block3 MATH500 Avg Step50=0.78525、100=0.75525、150=0.78025、200=0.754；终点并非各benchmark最好，保留全部里程碑。
- 两臂共享新prompt/独立request seeds及原始初始化；相较旧ChatML同时变化多个协议因素，不能由新绝对分数证明旧失败全由单一因素或Block3普适有效。

**缺失与混淆：**

- 匹配completion原始Base完整benchmark分数未找到。
- 新Token50/100/150/200完整benchmark分数等待父线程；所有缺失显式null，不用历史1.7/0.6 Token代替。
- 新协议公共GRPO教师完整四benchmark分数未找到；16题4/32与8/32不是held-out benchmark。

### historical external

| 方法/对象 | Step | MATH500 | AIME24 | AIME25 | AMC23 | 记录/来源 |
|---|---:|---:|---:|---:|---:|---|
| block3_mean | 50 | 78.53 / 91.80 | 14.17 / 30.00 | 11.67 / 23.33 | 50.30 / 74.70 | E157；[S185](#s185) |
| block3_mean | 100 | 75.52 / 90.80 | 15.00 / 30.00 | 12.08 / 33.33 | 47.29 / 74.70 | E158；[S188](#s188) |
| block3_mean | 150 | 78.03 / 91.20 | 15.42 / 33.33 | 11.67 / 26.67 | 49.70 / 77.11 | E159；[S191](#s191) |
| block3_mean | 200 | 75.40 / 90.60 | 13.33 / 33.33 | 13.75 / 30.00 | 46.23 / 74.70 | E160；[S194](#s194) |
| token_opd | 50 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | E161；[S063](#s063) |
| token_opd | 100 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | E162；[S063](#s063) |
| token_opd | 150 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | E163；[S063](#s063) |
| token_opd | 200 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | E164；[S063](#s063) |
| student_base | 0 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | 缺失 / 缺失 | E165；[S063](#s063) |

**系列证据：** [S018](#s018)、[S060](#s060)、[S061](#s061)、[S062](#s062)、[S063](#s063)。

## 诊断不是Benchmark

下列推理诊断全部使用已选训练题，各prompt、长度上限、请求seed规则分开。不是新的训练seed，不计入上面的held-out分数；原始正确/截断计数与raw哈希见JSON `diagnostic_evidence`。

### llama_prompt_16

来源：[S223](#s223)。

| 模型 | Prompt cell | 输出数 | 判对数 | Length stop |
|---|---|---:|---:|---:|
| student | historical | 32 | 0 | 1 |
| student | no_think | 32 | 1 | 1 |
| student | completion | 32 | 1 | 4 |
| student | eval | 32 | 0 | 1 |
| teacher | historical | 32 | 3 | 2 |
| teacher | no_think | 32 | 2 | 4 |
| teacher | completion | 32 | 1 | 4 |
| teacher | eval | 32 | 2 | 0 |

- 原生chat不是Base套ChatML；去chat的completion反而使学生截断1/32->4/32，不证明去chat可修复Llama。

### qwen4_completion_prompt_16

来源：[S224](#s224)。

| 模型 | Prompt cell | 输出数 | 判对数 | Length stop |
|---|---|---:|---:|---:|
| student | historical | 32 | 1 | 18 |
| student | plain | 32 | 2 | 0 |
| student | candidate | 32 | 4 | 1 |
| teacher | historical | 32 | 6 | 2 |
| teacher | plain | 32 | 5 | 0 |
| teacher | candidate | 32 | 8 | 3 |

- 同16题candidate学生4/32/教师8/32，不是MATH500或teacher可靠性上界。
- 所有prompt cell单列，不取最优cell与正式结果拼接。

### qwen06_qwen17_truncation_192

来源：[S225](#s225)。

| 32输出诊断单元 | 判对 | Length stop |
|---|---:|---:|
| q06_base | 0 | 12 |
| q06_step50 | 1 | 10 |
| q17_base | 1 | 16 |
| q17_step50 | 4 | 4 |

其余first16 prompt ablation数值完整保留在JSON，子集不与primary32相加。

- primary32与first16子集有重叠，不能相加称更多独立样本。
- no_think仅删指令，未显式false；不同于后续math_eval_nonthinking_v1。

### qwen4_initial_loops

来源：[S058](#s058)。

| Prompt | 请求seeds | 4B截断/32 | 1.7B截断/32 |
|---|---|---:|---:|
| historical_chatml | fixed21 | 8 | 0 |
| historical_chatml | 21..28 | 8 | 8 |
| remove_think_only | 21..28 | 13 | 11 |
| remove_3_chatml_markers_only | fixed21 | 8 | 0 |
| remove_3_chatml_markers_only | 21..28 | 4 | 2 |
| plain_solution | 21..28 | 2 | 1 |

- 独立原始模型vLLM复核，非正式training首批16/32；1024短筛与16384cell不合并。
- marker干预并非新candidate completion正式评测。HF tail与CPU first-token证据见原JSON，不能解释成已证明初始化历史。

### completion_update_resume_gate

来源：[S226](#s226)。

- 两臂各原始->Step1保存退出->恢复Step2，4进程exit0；每批32独立request seed、每题8不同输出。
- first-rollout输入/输出token/mask相同，logprob仅28/32逐位相同，max diff0.0142758；不是bit-exact恢复保证。
- Block3 Step1 cap0/32、Step2 cap2/32；Token Step1 cap0/32、Step2 cap4/32，不证明长期无循环。

## 多版本裁决

- **early_weights50/early_outcome200：** 按原preds_jsonl/checkpoint区分，primary以regrade_train/regrade_ml2各条原JSON优先；robust_regrade_curve与early_stopping_curve只是跨run拼接摘要，不生成额外训练复现。
- **early_blocksum50/early_blockadv50：** 两次独立50step run。0.4685与0.4655是不同naive Block3，不是互相纠错；仅MD/HTML有值，保留舍入不造原始计数。
- **qwen17_ml2：** final audited external summary优先于早期单Block3报告；external/builtin分开，3 checkpoint每个任务与finalization comparison数值完全一致。
- **qwen17_sweep/qwen17_ml2：** 旧train无显式eval seed且内存微批不同，不能把旧train较高绝对分数作为第二个严格同协议复现，也不能把ML2较低归因硬件。
- **qwen17_reeval：** 9月重生成保留独立generation_batch；不替代7月原评，不算额外training seed。
- **window_random_sliding：** 配置/总HTML的pending以及recovery子队列failed为中间快照；root queue complete与final external per-step summaries优先。canonical symlink实际落点记录于sources.resolved_path。
- **qwen06_nonthinking：** 9月17日MD的Block3尚未评测已被9月18日v2完整acceptance/queue complete取代；MD过程质量警报仍保留。
- **llama_historical：** v2父queue failed是后续Ray probe失败，不否定先前Qwen6eval/Llama Base完成；v4已complete。v1旧Base与v2复用Base是两批生成，数值不同都留。
- **qwen4_completion：** 本地progress completed_steps=[200]是旧快照；canonical四checkpoint acceptance均通过，Step200 SHA与本地记录一致。Token训练complete不等于Token新评测complete。
- **all：** 只保留真实已做实验：相关工作HTML中的外部论文表、候选baseline、seeds22/23以及未来权重sweep不是本项目已做结果，不计实验总数。

## 论文诚实边界

- 历史主张最多是条件相关的credit-assignment现象与失稳边界，尚非跨模型稳健提升，更非teacher计算显著减少。
- 新completion主表必须先补齐同prompt、同grader、同采样协议的Base与Token对照，再报告所有预定checkpoint或预先固定终点。当前不补写“优于Token”。
- 旧ChatML失败、0.6B/ Llama负面、DeepSeek未复现、random/sliding后期退化、中止和环境失败均应保留在正文限制或附录，不能按成功筛选历史。
- 所有单seed配对均不覆盖训练方差；历史macro置信区间不能搬作AIME/其他单任务CI。AIME每项仅30题，应避免夸大微小差异。
- 相关工作表格及未执行baseline/额外seed只属于计划，不能写成项目已做实验。本账本不代替论文最终的主表取舍。

## 核验

已逐字段比对 676 个精确分数单元；12 个文档舍入单元降级D；88 个缺失指标显式null。四份本地最终化配对JSON哈希匹配。现代per-eval acceptance与summary各任务分数一致；所有18组配对差值均由本账本同协议原值计算。

没有重新执行训练、推理、判分或完整raw重验；因此不声称独立复算所有原始正确题数。文件结构检查与引用/数值一致性是文档审计验证，不是实验代码测试。

## 来源索引

下面A/B/C/D/P遵循本文开头的可信度定义。远端路径只作出处，不表示需要或允许连接train。每项完整SHA256、字节数、canonical symlink落点均在JSON `sources`；模型资产本体哈希是原manifest登记值，本次未重新读取大权重。

<a id="s001"></a>
- **S001 [P]** [configs/experiments/block_opd/2026-07-08-block-advantage.yaml](../../../configs/experiments/block_opd/2026-07-08-block-advantage.yaml)；SHA256 `776b9e38ef1357d48e81d58c43f512daba8df5266ec4d9737849f2baa62adfda`。
<a id="s002"></a>
- **S002 [D]** [docs/results/2026-07-08-block-advantage.md](../../../docs/results/2026-07-08-block-advantage.md)；SHA256 `542484754909f24419ef44026c06c630626aae48f995098f84cd52bba7c23a5f`。
<a id="s003"></a>
- **S003 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/run_card.json`；SHA256 `34ef46a97691771f9666a29703f2dbfc963d3de5523bc92e799b0301ae31024d`。
<a id="s004"></a>
- **S004 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/teacher/model_identity.json`；SHA256 `3903eb5c50ff4cd18e3df8e30716e6f2976e28756cd299c741cd0c507d24a622`。
<a id="s005"></a>
- **S005 [P]** [docs/plans/2026-07-12-deepseek-justrl-paired-validation.md](../../../docs/plans/2026-07-12-deepseek-justrl-paired-validation.md)；SHA256 `92f649a104e9b5c4db1d9d5d3a74db22d949145c9a45213245db69f9e1120738`。
<a id="s006"></a>
- **S006 [B]** [reports/paired_validation/deepseek_justrl/comparisons/external.json](../../../reports/paired_validation/deepseek_justrl/comparisons/external.json)；SHA256 `6fb54b269aeb716a7f09d92bc0dc188ce293ddfd254c13a9e547c109bdf1a219`。
<a id="s007"></a>
- **S007 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-0.6B-Base/asset_manifest.json`；SHA256 `438fcb1041583b1fe59097214e947298cd8b27261a3d2376d76e383ef66533f4`。
<a id="s008"></a>
- **S008 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base/asset_manifest.json`；SHA256 `142fb615bb8188144610698c0a2e18402f5ca690412ea85bd2c785d48f6159d9`。
<a id="s009"></a>
- **S009 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Llama-3.2-1B-Instruct/asset_manifest.json`；SHA256 `323b2871960951f7608a114a2ddc482ce1ba1684c7168f5f8cb2f41d39ded8d0`。
<a id="s010"></a>
- **S010 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Llama-3.2-3B-Instruct/asset_manifest.json`；SHA256 `11b0643924c36cc45eb3c34792bb923f1f4d5afb408632ef3bace8d02b09ad92`。
<a id="s011"></a>
- **S011 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_cleanroom_20260707/data/public_math/manifest.json`；SHA256 `94bb080164e3d1666fb9d23dd77b7214e5a4505e99cd713e8aded8a60b5a5823`。
<a id="s012"></a>
- **S012 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_cleanroom_20260707/runs/eval_gsm8k_outcome200_20260707_192132/seed7_standard/summary.json`；SHA256 `7e862d644d342d1028677819ed5c52f9d27d5e4b2a2aa82ba9088c9441d538bf`。
<a id="s013"></a>
- **S013 [P]** [docs/plans/2026-07-12-ml2-token-opd-paired-validation.md](../../../docs/plans/2026-07-12-ml2-token-opd-paired-validation.md)；SHA256 `eecbdac773588bb9787399f4c20dc2c338c41c349441e5baf050826a4c5f9845`。
<a id="s014"></a>
- **S014 [D]** [docs/results/2026-09-13-token-truncation-comparison.md](../../../docs/results/2026-09-13-token-truncation-comparison.md)；SHA256 `db919d84d53a74db627621176eba77f6b6ddf1c37615d359d978a8f5aed74c99`。
<a id="s015"></a>
- **S015 [P]** [docs/plans/2026-09-13-qwen06-nonthinking-token.md](../../../docs/plans/2026-09-13-qwen06-nonthinking-token.md)；SHA256 `911f00b577b5e44f912773c55dce51357b87dd6b5bb7108222c9907d80013877`。
<a id="s016"></a>
- **S016 [P]** [docs/plans/2026-09-18-llama32-paired-validation.md](../../../docs/plans/2026-09-18-llama32-paired-validation.md)；SHA256 `76a91cfe77912846527c781d76147af8cabcc422a6916e1cf255955311b869bf`。
<a id="s017"></a>
- **S017 [D]** [docs/results/2026-09-19-llama-historical17-supervision.md](../../../docs/results/2026-09-19-llama-historical17-supervision.md)；SHA256 `e6e6363c8f6fa66915532771fb748b850bac4dc5e1f8ddf1736409c30d63e29e`。
<a id="s018"></a>
- **S018 [D]** [docs/results/2026-09-20-qwen-completion-acceptance.md](../../../docs/results/2026-09-20-qwen-completion-acceptance.md)；SHA256 `53503d1622ad9adeab44100d723773dbb15ff32d7c271f87ed397944dea5a6f6`。
<a id="s019"></a>
- **S019 [C]** [results/pilot50_training_summary_all_variants.json](../../../results/pilot50_training_summary_all_variants.json)；SHA256 `d485b9861e9a66297c515605c6b261806de1198f281e425a2bc1f1ecd928a3e8`。
<a id="s020"></a>
- **S020 [C]** [results/ablation50_training_summary.json](../../../results/ablation50_training_summary.json)；SHA256 `d52480139adcfe894e34e2e946085668af0c7e225c9530ad5ef54e4475303980`。
<a id="s021"></a>
- **S021 [C]** [results/pilot50_math100_summary_table.json](../../../results/pilot50_math100_summary_table.json)；SHA256 `c4808f00164420a7dd341287d52b3b9c9c89c610d67ebaa0844b877ce0e5b830`。
<a id="s022"></a>
- **S022 [C]** [results/regrade_train_summary_table.json](../../../results/regrade_train_summary_table.json)；SHA256 `876cc06f9aba465c6a17bac42a74e0b4f05f4d4f9613a80a745c496fd0229af3`。
<a id="s023"></a>
- **S023 [C]** [results/outcome200_training_summary.json](../../../results/outcome200_training_summary.json)；SHA256 `f8f80c9775846986445cd59bd9d413d3072a8257c6f1dd918f9eddbfdca16a2c`。
<a id="s024"></a>
- **S024 [C]** [results/regrade_ml2_summary_table.json](../../../results/regrade_ml2_summary_table.json)；SHA256 `0c0235c319c328d720eed91ea554b39ed251645bd5e37ed35c20e02e9cc2c305`。
<a id="s025"></a>
- **S025 [C]** [results/curve_math500_std_topk_summary_table.json](../../../results/curve_math500_std_topk_summary_table.json)；SHA256 `6103c54e0509f61805e041252b4df5a26a2bb9be3900de3453da4845a4237a86`。
<a id="s026"></a>
- **S026 [C]** [results/robust_regrade_curve_summary.json](../../../results/robust_regrade_curve_summary.json)；SHA256 `7327e791fbb7487cafb5e0370429c72d1d249bf30885192e0e38a125f00c3ac3`。
<a id="s027"></a>
- **S027 [C]** [results/anchor100_training_summary.json](../../../results/anchor100_training_summary.json)；SHA256 `f823c058c37bca977e927581228a51741fd95cb27aeacc10d6fa2c34f22fa8c3`。
<a id="s028"></a>
- **S028 [C]** [results/anchor100_regrade_summary_table.json](../../../results/anchor100_regrade_summary_table.json)；SHA256 `874cd929ca5575ac4fe6f4cbf7e98f8b69fccd44dd3a3622dce4f1a0fb8a128b`。
<a id="s029"></a>
- **S029 [D]** [reports/block_opd_experiment_report.html](../../../reports/block_opd_experiment_report.html)；SHA256 `82e3279cca665148602bc8e7265b90c265a02b99432cf5e80ab7bf3643faf5e6`。
<a id="s030"></a>
- **S030 [C]** [results/2026-07-10-block-size-sweep-step200-n8.json](../../../results/2026-07-10-block-size-sweep-step200-n8.json)；SHA256 `fbb3913e16eb0a115e46a270baefc36a565fffc6a3139537b886ad4ce056ac36`。
<a id="s031"></a>
- **S031 [C]** [results/2026-07-12-block3-ml2-replication.json](../../../results/2026-07-12-block3-ml2-replication.json)；SHA256 `8ffd652f8b487ecf51dc2109849a27907f93fc833494ca79b1515e6a7f1e9961`。
<a id="s032"></a>
- **S032 [C]** [results/2026-07-11-block10-collapse-dual-host.json](../../../results/2026-07-11-block10-collapse-dual-host.json)；SHA256 `085aca9da2c315666a977e1a4a0c29480d69568b2aa4ef443cc7deb342e9e636`。
<a id="s033"></a>
- **S033 [D]** [reports/block10_collapse_analysis.html](../../../reports/block10_collapse_analysis.html)；SHA256 `24142e76cbb0205f293394e06dc09bdecf2e253bf583eb51d94f84d91fc4765d`。
<a id="s034"></a>
- **S034 [B]** [reports/paired_validation/ml2/comparisons/external.json](../../../reports/paired_validation/ml2/comparisons/external.json)；SHA256 `3f3b6f3384e3b02b736fae65e6ff2ec1d1ebcba964f716f2d3064eb46a66b074`。
<a id="s035"></a>
- **S035 [B]** [reports/paired_validation/ml2/comparisons/builtin.json](../../../reports/paired_validation/ml2/comparisons/builtin.json)；SHA256 `d4c64ffe448bac89bde475801542b61c88e2c2a41e38f07d5030fde38a86b7e9`。
<a id="s036"></a>
- **S036 [P]** [reports/paired_validation/ml2/finalization_hashes.sha256](../../../reports/paired_validation/ml2/finalization_hashes.sha256)；SHA256 `1592f83be03d43b48f1a295e6b67af28560f5dbd42718330f2f80f719127355b`。
<a id="s037"></a>
- **S037 [D]** [docs/results/2026-09-18-historical17-reeval-startup.md](../../../docs/results/2026-09-18-historical17-reeval-startup.md)；SHA256 `475d06024bbbe59e7cbabe2500ddf5e9f3d5a6285c40090db54f691460e78c58`。
<a id="s038"></a>
- **S038 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/acceptance.json`；SHA256 `51563bd2c8d1bcfb6e485ff62c15ed9da13ce5f41a59f2d49b6630a29fe4f448`。
<a id="s039"></a>
- **S039 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/comparison.json`；SHA256 `8833d03178dff92bca3226279d9dcbb28622f32206922024cad1d2dea6cf474f`。
<a id="s040"></a>
- **S040 [B]** [reports/paired_validation/deepseek_justrl/comparisons/builtin.json](../../../reports/paired_validation/deepseek_justrl/comparisons/builtin.json)；SHA256 `3fa0dc0eddb97e6e8071fc41e27ba7d78fe4527cafc3d04584d163eb4edccb6c`。
<a id="s041"></a>
- **S041 [P]** [reports/paired_validation/deepseek_justrl/finalization_hashes.sha256](../../../reports/paired_validation/deepseek_justrl/finalization_hashes.sha256)；SHA256 `208c7454dbf62e1f9681ee344b303ca6156a06474bae59b440fb020de0d9df9c`。
<a id="s042"></a>
- **S042 [C]** [results/2026-07-13-block3-cross-pair-validation.json](../../../results/2026-07-13-block3-cross-pair-validation.json)；SHA256 `97e738312150d261c357a57aff2fe36beac2c89a7260f5ee5f1e4f17a94e7d7e`。
<a id="s043"></a>
- **S043 [P]** [configs/experiments/revisiting_opd/sliding_window_validation.yaml](../../../configs/experiments/revisiting_opd/sliding_window_validation.yaml)；SHA256 `771323e90e8ef2a589cd765c2e2bff28e67b369dcf1123ae2da1d5cf98c8f84e`。
<a id="s044"></a>
- **S044 [D]** [docs/results/2026-09-12-window-eval-recovery.md](../../../docs/results/2026-09-12-window-eval-recovery.md)；SHA256 `d654556d68ec21f5d5ae67e46ab72b657410253f7b49ad7c2ea8fb7958adf202`。
<a id="s045"></a>
- **S045 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/paired_comparison.json`；SHA256 `355bc88a6669c82ce9f3771fb8a3b86fb1816a98ca2b189d48882f97ccb5e25e`。
<a id="s046"></a>
- **S046 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/queue_state.json`；SHA256 `ebfcf28bb59fa4926ef1a1214e079e98131a549a30eef5300d5b0e271a973696`。
<a id="s047"></a>
- **S047 [D]** [docs/results/2026-09-13-qwen06-truncation-diagnosis.md](../../../docs/results/2026-09-13-qwen06-truncation-diagnosis.md)；SHA256 `e13c274f348aa2eb38e68973fdc28c78fe0375de25fca404b5015cfd8fa5ffa8`。
<a id="s048"></a>
- **S048 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v1_qwen06_pair_seed21_ml2/queue_state.json`；SHA256 `217b9a5a4864fef31d2fad90b7e507a16ca3f4cce7c91d09823bbcb664a68c9e`。
<a id="s049"></a>
- **S049 [D]** [docs/results/2026-09-17-qwen06-evaluation-progress.md](../../../docs/results/2026-09-17-qwen06-evaluation-progress.md)；SHA256 `43bad396407f4c98e37931c358241f04f539ed5e6de29b9d36cbf3789045d126`。
<a id="s050"></a>
- **S050 [D]** [docs/results/2026-09-17-qwen06-block3-eval-startup.md](../../../docs/results/2026-09-17-qwen06-block3-eval-startup.md)；SHA256 `92a7e46e7dc951f00f26edfbe36eeaf04b7f5b2cda3629853802977728134a54`。
<a id="s051"></a>
- **S051 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2/block3_comparison.json`；SHA256 `638e946c0b626cc263937a119002fa5d66b4f1f72a902a9a2193fe0ae1ba73ab`。
<a id="s052"></a>
- **S052 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v1_llama32_1b_3b_nonthinking_seed21_ml2/queue_state.json`；SHA256 `79f4694bbd77d82b303e618256f4d79add831cd5032394344b4343321897ba09`。
<a id="s053"></a>
- **S053 [C]** [docs/results/2026-09-19-llama-historical-prompt-audit.json](../../../docs/results/2026-09-19-llama-historical-prompt-audit.json)；SHA256 `42fbee6fe595cb078ec919b20ffec0c967a04272738688e9a69165c6b7049101`。
<a id="s054"></a>
- **S054 [D]** [docs/results/2026-09-19-llama-block3-quality-diagnosis.md](../../../docs/results/2026-09-19-llama-block3-quality-diagnosis.md)；SHA256 `1044abf38c137b860d279993943e688e0bc837fe08869c858c4899dd82407c67`。
<a id="s055"></a>
- **S055 [D]** [docs/results/2026-09-18-llama-historical17-recovery.md](../../../docs/results/2026-09-18-llama-historical17-recovery.md)；SHA256 `f7588cce9b9aa97d314e8185e66c274446a8dfc583adc2f84dbf606ea1e189cc`。
<a id="s056"></a>
- **S056 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/paired_comparison.json`；SHA256 `d3ff5541ff33a5564062af5e633098d35042c2183b1120bbec382b680bbc7ae3`。
<a id="s057"></a>
- **S057 [D]** [docs/results/2026-09-19-qwen4-initial-loop-incident.md](../../../docs/results/2026-09-19-qwen4-initial-loop-incident.md)；SHA256 `561ee46642eb7e78cabf618850ec69b98c0d057ab6a8bfd8d4294aa6295b8af2`。
<a id="s058"></a>
- **S058 [C]** [docs/results/2026-09-19-qwen4-initial-loop-summary.json](../../../docs/results/2026-09-19-qwen4-initial-loop-summary.json)；SHA256 `96645b9be3eef34c67d3856a33b79b795550ad5d4230b6176dc3beb088caca07`。
<a id="s059"></a>
- **S059 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260919v1_qwen4_blockfirst_seed21_ml2/queue_state.json`；SHA256 `8d515d83b392c21fecdeea2defe090535c5a3e314177281c827f8bae85c8df25`。
<a id="s060"></a>
- **S060 [C]** [docs/results/2026-09-20-qwen4-completion-block3-launch-v2-failed.json](../../../docs/results/2026-09-20-qwen4-completion-block3-launch-v2-failed.json)；SHA256 `375a6d1ae830df6a082b7f49db0c650ef8f12184b5f2d8f78121b93d576225b1`。
<a id="s061"></a>
- **S061 [C]** [docs/results/2026-09-20-qwen4-completion-block3-launch-v3.json](../../../docs/results/2026-09-20-qwen4-completion-block3-launch-v3.json)；SHA256 `96d3449bd529f6872f8140b03a9d84cc2648af3184c87378c830a361ff4b814e`。
<a id="s062"></a>
- **S062 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/queue_state.json`；SHA256 `4075a3b781fefd51974b803b1ca92a537f6cb6237e6de7b794e34df2b10fd245`。
<a id="s063"></a>
- **S063 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_token_seed21_ml2/queue_state.json`；SHA256 `1f70bcacffab13c0d6a57b48142cce203eb11c73b589ec57c81441d335a55766`。
<a id="s064"></a>
- **S064 [C]** [results/pilot50_gsm100_summary_table.json](../../../results/pilot50_gsm100_summary_table.json)；SHA256 `961a40536d60e65de2dd2f5bb90c0346212883a3130df2a53f8c2908c47e731e`。
<a id="s065"></a>
- **S065 [C]** [results/ablation50_gsm100_summary_table.json](../../../results/ablation50_gsm100_summary_table.json)；SHA256 `e56a794a900c06353f307bd53b284ecc8e9a315fdf6ab863aee5406c458fa321`。
<a id="s066"></a>
- **S066 [C]** [results/ablation50_math100_summary_table.json](../../../results/ablation50_math100_summary_table.json)；SHA256 `90f8df5653e7fec3e246fb454b95633d2b8438ad9eb27e01e7df4d66940dc35e`。
<a id="s067"></a>
- **S067 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_cleanroom_20260707/runs/regrade_ml2_20260707_203516/gsm8k_seed7_outcome_reweight.json`；SHA256 `c4c5eeaec2e48b3c56d427ddbe2a0b555eba4f388313b2619f959426ecb789d3`。
<a id="s068"></a>
- **S068 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_cleanroom_20260707/runs/regrade_ml2_20260707_203516/gsm8k_seed7_outcome_topk_router.json`；SHA256 `a6122b901004727869d44197ea6c5d6d30035c0b7ff6ae543841d1cc47c0789f`。
<a id="s069"></a>
- **S069 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_cleanroom_20260707/runs/regrade_ml2_20260707_203516/gsm8k_seed7_standard.json`；SHA256 `6737a07e5643923b73b438b594d9ca82fba6581dc8a1d68d2adbb757495ca9b9`。
<a id="s070"></a>
- **S070 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_cleanroom_20260707/runs/regrade_ml2_20260707_203516/gsm8k_seed7_topk_router.json`；SHA256 `24f12cd4e1a85daf67c3119b5acf977d6b69e4d82b0839bd65e69e57acb3d83d`。
<a id="s071"></a>
- **S071 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_cleanroom_20260707/runs/regrade_ml2_20260707_203516/math500_seed7_outcome_reweight.json`；SHA256 `4e8e604dddc07092fb4d2501970b37d3dba03ba980fb018f809499ec1eb5066c`。
<a id="s072"></a>
- **S072 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_cleanroom_20260707/runs/regrade_ml2_20260707_203516/math500_seed7_outcome_topk_router.json`；SHA256 `1059e334902cb9b44d96540795f5b9ef5a464e834614cddbe9cf2023aa506420`。
<a id="s073"></a>
- **S073 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_cleanroom_20260707/runs/regrade_ml2_20260707_203516/math500_seed7_standard.json`；SHA256 `25cd51c6b36dda992cc283c9149d0b7f207fbbcef5c6a63533743486c3b30bb8`。
<a id="s074"></a>
- **S074 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_cleanroom_20260707/runs/regrade_ml2_20260707_203516/math500_seed7_topk_router.json`；SHA256 `6e0d76240919d96754b98374da41223037a563a4a3ef49bea2233f28cfb9a087`。
<a id="s075"></a>
- **S075 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260708_block3_dapo17k_paper_qwen3/token_opd/eval_step_200_n8/outputs/summary.json`；SHA256 `f58e188056dd7bf0babdf3d53785c0baf445d8d7143233456f8f3b94b59fbb7d`。
<a id="s076"></a>
- **S076 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260708_block3_dapo17k_paper_qwen3/block3_mean/eval_step_200_n8/outputs/summary.json`；SHA256 `92e4728733cd59dac478223519b2f470ca0ded8d291ff704b753c29d3ef46051`。
<a id="s077"></a>
- **S077 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260709_block10_dapo17k_paper_qwen3/block10_mean/eval_step_200_n8/outputs/summary.json`；SHA256 `df5b51f9dc3687dbd230237bdcb2bb655cd8d146250ae0ad4c5921b871132641`。
<a id="s078"></a>
- **S078 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260710v5_block10_collapse_diag_seed21_ml2/block10_mean/eval_step_50_n8/outputs/summary.json`；SHA256 `d1ed16c80749f6beb354c1b05ee420ba2b61b4f04146a75470fa83cc201f2076`。
<a id="s079"></a>
- **S079 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260710v5_block10_collapse_diag_seed21_ml2/block10_mean/eval_step_100_n8/outputs/summary.json`；SHA256 `fbdee2a19a7d22cec0b33748924eec99485cc9ceab46d97e8545bbaf13b5879c`。
<a id="s080"></a>
- **S080 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260710v5_block10_collapse_diag_seed21_ml2/block10_mean/eval_step_200_n8/outputs/summary.json`；SHA256 `1324c32deb05c76bd4f8986346fb6e9ceb31f4cd39bdf20ffd9574ed19c692da`。
<a id="s081"></a>
- **S081 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/eval_step_50_n8/outputs/eval_config.json`；SHA256 `c573965217ac799e120e49d40b5eeb4eed55e0f56bf7b53a944f6b8c436bf354`。
<a id="s082"></a>
- **S082 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/eval_step_50_n8/outputs/summary.json`；SHA256 `86634f054b909e8c352c1deedf7beb4ac07663312aee04ea6513d3b273fc6374`。
<a id="s083"></a>
- **S083 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/eval_step_50_n8/historical_external_grader_audited/summary.json`；SHA256 `dbec6904c5d487039b8bc857be836846c43dbaa86a3f00311ce3dd5406c3722e`。
<a id="s084"></a>
- **S084 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/eval_step_100_n8/outputs/eval_config.json`；SHA256 `34bda7b42f65ed672e8967d6845ae79e69fcff1c86d1cd0f44b7a2831c255c33`。
<a id="s085"></a>
- **S085 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/eval_step_100_n8/outputs/summary.json`；SHA256 `ada6d73bc1ca980bba09015866bc4514762ffa2396cfa045665f857aafecac3b`。
<a id="s086"></a>
- **S086 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/eval_step_100_n8/historical_external_grader_audited/summary.json`；SHA256 `31153267cd37691b670447cd1874e06f0f42d084d0c2845ec2f9fb7bff29eff4`。
<a id="s087"></a>
- **S087 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/eval_step_200_n8/outputs/eval_config.json`；SHA256 `c11d53d362814cede498a01073e44d6241923cd9d241e53ea35da44685b8a074`。
<a id="s088"></a>
- **S088 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/eval_step_200_n8/outputs/summary.json`；SHA256 `336d3b52c7468a1548a4c18b32beb56896405fba026cc80ea108b94b7604efbc`。
<a id="s089"></a>
- **S089 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/eval_step_200_n8/historical_external_grader_audited/summary.json`；SHA256 `deb50bd4233e434d5046aeeb5a49d731c72e9d4c886f48f369a01ee54e660dd6`。
<a id="s090"></a>
- **S090 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean/eval_step_50_n8/outputs/eval_config.json`；SHA256 `893210c8c81bffafa650a0d39d53c2c190cc150c093b7174c0a5f3a15e417f58`。
<a id="s091"></a>
- **S091 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean/eval_step_50_n8/outputs/summary.json`；SHA256 `33b0b267fa727d1b8b13d0b496e6e27c735410f01407a0ddebf2da227f5acceb`。
<a id="s092"></a>
- **S092 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean/eval_step_50_n8/historical_external_grader_audited/summary.json`；SHA256 `5bdd9710f1a5e92c009ebdf6303eb22a28601417ec5aa2fbe56c3c954e5aaad3`。
<a id="s093"></a>
- **S093 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean/eval_step_100_n8/outputs/eval_config.json`；SHA256 `23683368f6d81e3097e389872d30ef42249f81b8ff43e47305eaebf799ec15e3`。
<a id="s094"></a>
- **S094 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean/eval_step_100_n8/outputs/summary.json`；SHA256 `0b7e4c198534fc9fd53212b961987dd1f24de34cf729e34efe24271d3546a53b`。
<a id="s095"></a>
- **S095 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean/eval_step_100_n8/historical_external_grader_audited/summary.json`；SHA256 `851dc4a69d7ce50d04aab2b8ea761b9af98b9805ee67e8d36cff3109e30c84f9`。
<a id="s096"></a>
- **S096 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean/eval_step_200_n8/outputs/eval_config.json`；SHA256 `335cd90b4bcb1c2fd2cf5f0c6f941c4154c110751aad7109e2aeea7117404d57`。
<a id="s097"></a>
- **S097 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean/eval_step_200_n8/outputs/summary.json`；SHA256 `3e7ab5bf57f2ed41df187df43e44a364a0d10908df2a47157904fc19bd9ee954`。
<a id="s098"></a>
- **S098 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean/eval_step_200_n8/historical_external_grader_audited/summary.json`；SHA256 `185c7c9842feb3d621c9e24ed971b1fcbca20e0fd7e94783c1c844ab017c4a60`。
<a id="s099"></a>
- **S099 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3/eval_step_50_n8/outputs/eval_config.json`；SHA256 `6af92e318ef87578ac009e903786d7028c4798133f530fafbf470644a0539024`。
<a id="s100"></a>
- **S100 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3/eval_step_50_n8/historical_external_grader/summary.json`；SHA256 `5a939f59cb48c743a47dd1fe16a938d04a6b3e16e2fd8c8e25a1d8dbf5dfdb0b`。
<a id="s101"></a>
- **S101 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3/eval_step_50_n8/outputs/summary.json`；SHA256 `9936c078e7f4a4435141a4d04b1d8d8e92fa191a230106c162b13b9d1d431402`。
<a id="s102"></a>
- **S102 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3/eval_step_100_n8/outputs/eval_config.json`；SHA256 `1cbb4b9099b313c270584552dd2fb2eb7e8ee4632b4cf6a1a94b888c9ef4da21`。
<a id="s103"></a>
- **S103 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3/eval_step_100_n8/historical_external_grader/summary.json`；SHA256 `16cedf9383eae3072e10d6219d046a45239d0645c2082ca3d92f1a551b6c0ded`。
<a id="s104"></a>
- **S104 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3/eval_step_100_n8/outputs/summary.json`；SHA256 `8377f783747c48f4ecad42a8be70d68501476ac827f926ab5124b7a4c2a81cda`。
<a id="s105"></a>
- **S105 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3/eval_step_200_n8/outputs/eval_config.json`；SHA256 `b39b29655e0333e42095cb48ef038d34c6ecd7626de59d2f1a1b713d11002daa`。
<a id="s106"></a>
- **S106 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3/eval_step_200_n8/historical_external_grader/summary.json`；SHA256 `cc22cc9f333eade083f019455c5ce8945ab7ea0d91ff6ebdfd1764ac04201715`。
<a id="s107"></a>
- **S107 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3/eval_step_200_n8/outputs/summary.json`；SHA256 `23a4cc2c80941ad54b1e086b02b13d0f36615d07f52a27a134a9a34967f5931a`。
<a id="s108"></a>
- **S108 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3/eval_step_50_n8/outputs/eval_config.json`；SHA256 `4e41e067ae48086819f488ac6ea0f9b81b09d8f122f6b9dc81af3ce29f83a691`。
<a id="s109"></a>
- **S109 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3/eval_step_50_n8/historical_external_grader/summary.json`；SHA256 `3db217181e0787ef299363123ea10d0e4de44694f18bb22306a440c647eba29f`。
<a id="s110"></a>
- **S110 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3/eval_step_50_n8/outputs/summary.json`；SHA256 `71fa7e4189338601aa4a5e7e0dc30acb8235e601a7058113c549dc9273a4e86d`。
<a id="s111"></a>
- **S111 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3/eval_step_100_n8/outputs/eval_config.json`；SHA256 `3e94bdff360fe602056255fac7b42dfa7e743394dc8cefa2e73cd0057b554a97`。
<a id="s112"></a>
- **S112 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3/eval_step_100_n8/historical_external_grader/summary.json`；SHA256 `0dd0333e3f38f77558e93c6621fdb6b66a1341c1848e0a37dff67c4a487f0863`。
<a id="s113"></a>
- **S113 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3/eval_step_100_n8/outputs/summary.json`；SHA256 `53ea518529f055ca286918161376bdff33b2b144467b7e027f65906990d8d156`。
<a id="s114"></a>
- **S114 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3/eval_step_200_n8/outputs/eval_config.json`；SHA256 `9e72bd6969eab68a90a5c69b13ae7be0e25b5378a5ea5dac14f36e0d7d809bce`。
<a id="s115"></a>
- **S115 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3/eval_step_200_n8/historical_external_grader/summary.json`；SHA256 `dea1d3fc2ff1ca8eb1a18d882d42c45b1b82272c2f763ca285a3d3727bdaa377`。
<a id="s116"></a>
- **S116 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3/eval_step_200_n8/outputs/summary.json`；SHA256 `6dfadecfbd8ded06f23067de7e28a4196fa40f60019de8b89f5c5de5fa3eecf7`。
<a id="s117"></a>
- **S117 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/token_opd_step50/outputs/summary.json`；SHA256 `c53ddb30ecad495ae3e9deae53ddcd512db0a2993b5b245729db010944dd8b3e`。
<a id="s118"></a>
- **S118 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/token_opd_step50/outputs/eval_config.json`；SHA256 `8df37e944d2f7c74b37be14c8ed67e96cf5503840913333bc49ed92a8f660149`。
<a id="s119"></a>
- **S119 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/token_opd_step50/acceptance.json`；SHA256 `1e6410c6b4bb269ed45672b1e79fcc415c9fb539b2e63a5b5612a4f1a2d8f5a7`。
<a id="s120"></a>
- **S120 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/token_opd_step100/outputs/summary.json`；SHA256 `38cf94e0724fa7bde3d1cfa53c0c3b6c2ec08199aaaea1401eb3cdd1fbc852e4`。
<a id="s121"></a>
- **S121 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/token_opd_step100/outputs/eval_config.json`；SHA256 `7c8a87287e013a80f8f1ee7de8a17f3577ae554cec9fc4a12ead0944498b9379`。
<a id="s122"></a>
- **S122 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/token_opd_step100/acceptance.json`；SHA256 `6e4f331a475852e9207f7049f000ef645e361ec69c201caf0d3e26926d096007`。
<a id="s123"></a>
- **S123 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/token_opd_step200/outputs/summary.json`；SHA256 `7ceefeca0732e697d5cc8ab9505b88b2a433249c394ebb03cab1cffb4f819eeb`。
<a id="s124"></a>
- **S124 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/token_opd_step200/outputs/eval_config.json`；SHA256 `4f05495f11bfb51d00d76fcf20550802ab580da9600e718f2b03c6cafa986bbe`。
<a id="s125"></a>
- **S125 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/token_opd_step200/acceptance.json`；SHA256 `61b7241bc110ee8a044334e6f2222286e949b777b45ca13eeddc883e8ba504e7`。
<a id="s126"></a>
- **S126 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/block3_mean_step50/outputs/summary.json`；SHA256 `245017a94ed220caa6b2958dea97f2f857b11d6c14d88b42acdc92bffff5b63a`。
<a id="s127"></a>
- **S127 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/block3_mean_step50/outputs/eval_config.json`；SHA256 `b2fc60cbbe5db5b6d28b036e6b7976f8323d4a0f87b2b18b50dac7986e89c3df`。
<a id="s128"></a>
- **S128 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/block3_mean_step50/acceptance.json`；SHA256 `ebc96ab97cf8765244e88289e603a3d9e29e05dcef6774beb320587e3e14505b`。
<a id="s129"></a>
- **S129 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/block3_mean_step100/outputs/summary.json`；SHA256 `d0236db92e07e3b7d6264603056352def9303181fe01b0eec61378b18989d199`。
<a id="s130"></a>
- **S130 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/block3_mean_step100/outputs/eval_config.json`；SHA256 `4a3ca46362b051f39dfb9aabed15704d6c4859b12f1116f9f9c9d88eb18a1870`。
<a id="s131"></a>
- **S131 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/block3_mean_step100/acceptance.json`；SHA256 `9d994370766072442ee38ec464609f3dc8db61b2707f9746e326baa2bf55fbd7`。
<a id="s132"></a>
- **S132 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/block3_mean_step200/outputs/summary.json`；SHA256 `5b42e371de95812b13d9c9a4e96cce74db4b68ad12b55fa55fbc0adb696ec8ca`。
<a id="s133"></a>
- **S133 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/block3_mean_step200/outputs/eval_config.json`；SHA256 `ee7abfab309a7ac38d3f24942fa3732bde4b8e76d32d94fc03879a6d8815430e`。
<a id="s134"></a>
- **S134 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/qwen17/evaluations/block3_mean_step200/acceptance.json`；SHA256 `d66b2f344630dbb7d775da89ce95787cc25c8587534e3cfa8a3f34dbe2ac2dd4`。
<a id="s135"></a>
- **S135 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/student_base/outputs/summary.json`；SHA256 `c411c2390889fbc53e8d2d37e02f425866acfcab50ed8c1dae23bf11946e2cb5`。
<a id="s136"></a>
- **S136 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/student_base/outputs/eval_config.json`；SHA256 `5bb03a3209ac93e05edd2fbbcde4a8abc9bd29eafe775527ed6d37d815bbb491`。
<a id="s137"></a>
- **S137 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/student_base/acceptance.json`；SHA256 `1c2cdf6d30e339e18e96441a18f468d3fb383a7516f8004246b934f18397e13c`。
<a id="s138"></a>
- **S138 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/teacher/outputs/summary.json`；SHA256 `352059852d861117ce543f69453c0fccd82aec89e39cbd65e1408d4d5ad3b9be`。
<a id="s139"></a>
- **S139 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/teacher/outputs/eval_config.json`；SHA256 `44f5677d0b2271c1d6cbe691a0e3886e133e53d6a5c659f6211274565625fe9c`。
<a id="s140"></a>
- **S140 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/teacher/acceptance.json`；SHA256 `fb4bfff07126279daaa494dfd0906aff858c04e07000fa6310b39a4254f6709b`。
<a id="s141"></a>
- **S141 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/token_step50/outputs/summary.json`；SHA256 `6f6d41afd40c7eacf358f6bf082fa74439a8230292e428450cb506fdf1d4ed39`。
<a id="s142"></a>
- **S142 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/token_step50/outputs/eval_config.json`；SHA256 `425b417a3ba8643cb171ee2d229b5865fc6a94ac96b0bd2d0bcdfd6dd97910dd`。
<a id="s143"></a>
- **S143 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/token_step50/acceptance.json`；SHA256 `5f4c7a623e49cfc53808b448d3b77af94d31a90a7059bbc7f08450ad41d09af0`。
<a id="s144"></a>
- **S144 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/token_step100/outputs/summary.json`；SHA256 `247bc9fed8cac817f3b0005db51a6c33a93524be1cfcc80c6f79550f8d86df3c`。
<a id="s145"></a>
- **S145 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/token_step100/outputs/eval_config.json`；SHA256 `d8043bd8249bd22f69e157c248b74cc502ace625b26099de000e533971c9cb73`。
<a id="s146"></a>
- **S146 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/token_step100/acceptance.json`；SHA256 `958114094564cf685a63f8eaf681472388f84e24d367730f9116932af6a7c2e3`。
<a id="s147"></a>
- **S147 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/token_step200/outputs/summary.json`；SHA256 `62b4772cc90a1774d3d1c67b8246e1ff03f4df775d0450eeffd0c14343b745fb`。
<a id="s148"></a>
- **S148 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/token_step200/outputs/eval_config.json`；SHA256 `7b9439bdab8d8300b23b41afe9fa19ef101037bb00451869f49a6cde360cc880`。
<a id="s149"></a>
- **S149 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/evaluations/token_step200/acceptance.json`；SHA256 `979bfa30311889324d92a5ebcceb58114001971096495a1051d4ec10f0a4747d`。
<a id="s150"></a>
- **S150 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2/evaluations/block3_step50/outputs/summary.json`；SHA256 `edee38ef85947d199982b36db2ba34bd3125ead5c3943fc680039d4a1dddab30`。
<a id="s151"></a>
- **S151 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2/evaluations/block3_step50/outputs/eval_config.json`；SHA256 `999ef13a2a362575f0f148e5bad531a8e7fcef73c1f610792f6b61dd697c4f6b`。
<a id="s152"></a>
- **S152 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2/evaluations/block3_step50/acceptance.json`；SHA256 `9f019ecac6dc8e3c8190f577e3a0590456d8d5e7babb2c39b75b653130108b4e`。
<a id="s153"></a>
- **S153 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2/evaluations/block3_step100/outputs/summary.json`；SHA256 `a56def8678cbb6f0912173d8545de4f3f2333700f3106ebca63b80f720b4750b`。
<a id="s154"></a>
- **S154 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2/evaluations/block3_step100/outputs/eval_config.json`；SHA256 `66e46020d35eaa208a32ad1239c7754a7d1a25c8abf84a6c514c3e9a6bf106a4`。
<a id="s155"></a>
- **S155 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2/evaluations/block3_step100/acceptance.json`；SHA256 `96fa156aa715dfa37b9b60c31e37469f7a9366c4fa6de2bc2b175c51dd50990e`。
<a id="s156"></a>
- **S156 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2/evaluations/block3_step200/outputs/summary.json`；SHA256 `37ac5788985541c3f704b31d9bda678188600d300d12e0661bd7d228a50da258`。
<a id="s157"></a>
- **S157 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2/evaluations/block3_step200/outputs/eval_config.json`；SHA256 `9b8aa372c82328837fe2072e0e72764f8149042c4a5f8ef2973b4fbe98ad2557`。
<a id="s158"></a>
- **S158 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2/evaluations/block3_step200/acceptance.json`；SHA256 `e333156ce4a22caf383ab893646ab2d9e6fd2afcc24f5cdd37d73ec48303dce2`。
<a id="s159"></a>
- **S159 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v1_llama32_1b_3b_nonthinking_seed21_ml2/evaluations/student_base/outputs/summary.json`；SHA256 `9ad3f1c7c68da912d52e66afa97cbad32ef69e2e57e5c5b0f11090581a789da3`。
<a id="s160"></a>
- **S160 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v1_llama32_1b_3b_nonthinking_seed21_ml2/evaluations/student_base/outputs/eval_config.json`；SHA256 `bd7a78ffd74abda01a2df4b54847d202655d4dd8ed11d6a09bd96d026a2caf9b`。
<a id="s161"></a>
- **S161 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v1_llama32_1b_3b_nonthinking_seed21_ml2/evaluations/student_base/acceptance.json`；SHA256 `71458336a7cff95762880b1b9dcc45f1131bf67e775cb49e6abd10407e27acea`。
<a id="s162"></a>
- **S162 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/llama32/evaluations/student_base/outputs/summary.json`；SHA256 `541981bbcdca19084448d42128829262b9c165e92f23db762550d720e0445715`。
<a id="s163"></a>
- **S163 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/llama32/evaluations/student_base/outputs/eval_config.json`；SHA256 `89993c7113a45e4a019ef07dd4bef19ab993b8293f0639b60e029bca6f517c2a`。
<a id="s164"></a>
- **S164 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/llama32/evaluations/student_base/acceptance.json`；SHA256 `310b0adc04fb5bdb73ee7dc80015b816a9b396423716ad78b006b5d2970c43fb`。
<a id="s165"></a>
- **S165 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/token_opd_step50/outputs/summary.json`；SHA256 `7d8696230aee9a1aa15853a2b39c7e35232f4b56259d8c90870517cbe1ba2f4b`。
<a id="s166"></a>
- **S166 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/token_opd_step50/outputs/eval_config.json`；SHA256 `ad07771cf34a243ec49d34b16e40b19b6f56f7afe576b4178894b5c203f2ad32`。
<a id="s167"></a>
- **S167 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/token_opd_step50/acceptance.json`；SHA256 `5b2bf09d8f50563f46cc12880fb04f651552e18f47edc60258fb43064d9ccd99`。
<a id="s168"></a>
- **S168 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/token_opd_step100/outputs/summary.json`；SHA256 `662192df7cf2d64ae449fe119a4a998345ca300ce5d9517b5e9afd6d3977f615`。
<a id="s169"></a>
- **S169 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/token_opd_step100/outputs/eval_config.json`；SHA256 `4ef33b1b801a9e63da1490e2b1232dc1a8e906cd5e944572ab2a1e49e4b31fb1`。
<a id="s170"></a>
- **S170 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/token_opd_step100/acceptance.json`；SHA256 `bfd4396d30cb082cc47bb3aa9b80fddcb34c27b38d1fa93fe8b265d21f1f4062`。
<a id="s171"></a>
- **S171 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/token_opd_step200/outputs/summary.json`；SHA256 `a778c8d5d514827585e6c3b3cd659970a3c7c90e296b8759751aac806d1fdde7`。
<a id="s172"></a>
- **S172 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/token_opd_step200/outputs/eval_config.json`；SHA256 `68da5b9ad2104ca0aea0604c451d02f726f337058b4d5a77e3afc6c3ca224cdc`。
<a id="s173"></a>
- **S173 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/token_opd_step200/acceptance.json`；SHA256 `018c553cc56ec3dfb4a44d3d204bebcba93f7bc315ee48eb45ba7f0740cf297e`。
<a id="s174"></a>
- **S174 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/block3_mean_step50/outputs/summary.json`；SHA256 `1a10657203ee3dc6c4d95a391d4548ebf992a041d1dbe7174e16a4fd3f8b70cb`。
<a id="s175"></a>
- **S175 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/block3_mean_step50/outputs/eval_config.json`；SHA256 `d9d1c546cab803ce523be662edc1b7f1fa09fd4fc22aa02f7a0c5635379c6488`。
<a id="s176"></a>
- **S176 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/block3_mean_step50/acceptance.json`；SHA256 `f04133bad0a48028276081fc69ee02b6b98693fd31e02b7fc9d21051ec954787`。
<a id="s177"></a>
- **S177 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/block3_mean_step100/outputs/summary.json`；SHA256 `a75cba0839c8eea6e8abfb67424426315943bd5c0c8a602fa727702089b43c8c`。
<a id="s178"></a>
- **S178 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/block3_mean_step100/outputs/eval_config.json`；SHA256 `80b733088a6a922c578c580c577ddca962e023a764e7b8d222ce73404e202381`。
<a id="s179"></a>
- **S179 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/block3_mean_step100/acceptance.json`；SHA256 `b8dec3e93e424f13e7ef9665ab390a2afe0bf17c169f6e83ad4b4b423ceb077b`。
<a id="s180"></a>
- **S180 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/block3_mean_step200/outputs/summary.json`；SHA256 `f8e24fc940c78a2be057ac6cfb5ec89473dadad86847fdac72efd2a1a00688aa`。
<a id="s181"></a>
- **S181 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/block3_mean_step200/outputs/eval_config.json`；SHA256 `9f5eb9e7125d16c5f59b23a637169b4eff21932d44e6ef97ba70864caa29d953`。
<a id="s182"></a>
- **S182 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/evaluations/block3_mean_step200/acceptance.json`；SHA256 `a5eb9cb638e357b01dc83f7b28c4540067afd21d4667e96fd661124a48ad379e`。
<a id="s183"></a>
- **S183 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step50/outputs/summary.json`；SHA256 `b04b009d9de1edc41a1766d62e34204e3c5c3e55145ce535b2fc33ef9600e53f`。
<a id="s184"></a>
- **S184 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step50/outputs/eval_config.json`；SHA256 `d24f3431357ea637306d289e2e6297e0c41f9a06ea62c392eb5abf4f981154cd`。
<a id="s185"></a>
- **S185 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step50/acceptance.json`；SHA256 `a21821fd400f8c02c3aab68587c4406f4c8208bcfe4d3cc7c46e0cf5ee4c32ab`。
<a id="s186"></a>
- **S186 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step100/outputs/summary.json`；SHA256 `cd0c610e16ede9dcbb79490230ae47121ec33e2b986be514e3880cdda7e1ebd0`。
<a id="s187"></a>
- **S187 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step100/outputs/eval_config.json`；SHA256 `6a11d9b00096d9490bbe8f5ec844cf33554718756e7a42ea3f66a4b9c085222f`。
<a id="s188"></a>
- **S188 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step100/acceptance.json`；SHA256 `0d3c40f5553cf63341585c0763aaeb1866305fb165abb23b23c497beafc7d263`。
<a id="s189"></a>
- **S189 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step150/outputs/summary.json`；SHA256 `8ee63a9a3175a2a72e8efe0f25f515bc94727aae79a27fb91835336d803edc5c`。
<a id="s190"></a>
- **S190 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step150/outputs/eval_config.json`；SHA256 `4850c9951f36c87145938b944b68690480c40039b3ebad117e9ee6540f1f6d4a`。
<a id="s191"></a>
- **S191 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step150/acceptance.json`；SHA256 `427cce92434d0967e9010d7a524124f48fe743cde033b9492197e6d87cea697c`。
<a id="s192"></a>
- **S192 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step200/outputs/summary.json`；SHA256 `704d1afa98ab13fe51d09f2e52e549dd9e00f1303767d17454c05898ae248b7a`。
<a id="s193"></a>
- **S193 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step200/outputs/eval_config.json`；SHA256 `7fa92fe2f4be44531244034e778cc0c81447b4eda648e802bea5775765754485`。
<a id="s194"></a>
- **S194 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2/evaluations/block3_mean_step200/acceptance.json`；SHA256 `0e2fe54f0686004c202888cbd556a5d2813eee3823c7956ceac665a985becfc1`。
<a id="s195"></a>
- **S195 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/acceptance.json`；SHA256 `ec6832ad6fd1871d150b1fc973e1f8aaad891b0f36cd129093c49f5b2756d3eb`。
<a id="s196"></a>
- **S196 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean/run_card.json`；SHA256 `fd656ae1d9c74a211a2e8e0ec683a96b7660b2a1968817ecab19a6191082932a`。
<a id="s197"></a>
- **S197 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260711v2_block3_replication_seed21_ml2/block3_mean/acceptance.json`；SHA256 `d9b4866986e4f5ccc9087a9da106bf703da459f8b6c5990963506fe0c22f5647`。
<a id="s198"></a>
- **S198 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260710v5_block10_collapse_diag_seed21_ml2/block10_mean/run_card.json`；SHA256 `2c6c99b8b41a8af4e861b9b14eecbb2213fc7b5b69dd1d9f9a1c1bbb4623295c`。
<a id="s199"></a>
- **S199 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260710v5_block10_collapse_diag_seed21_ml2/block10_mean/acceptance.json`；SHA256 `75de82d370dab9dc8e439801353f83867d83d29eccb4c0ccf42ffa44d702ba18`。
<a id="s200"></a>
- **S200 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3/run_card.json`；SHA256 `1f366d940f3b5f4a4a784ff7d31bf31fc1f08c41f927177d31b1c0209c7fcbe0`。
<a id="s201"></a>
- **S201 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/random3/acceptance.json`；SHA256 `99da90353e18d5fac755f32608a1a29edfac57721c2b322ad0490787e9c81ca2`。
<a id="s202"></a>
- **S202 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3/run_card.json`；SHA256 `600c551631e1ab727f807ae6ee4633a1ca2682ef1c301071a3fe9f5f4cfd4ec0`。
<a id="s203"></a>
- **S203 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/sliding3/acceptance.json`；SHA256 `220a9800762fef2edf6803a1042a9841e4745c7b395937e9941d752587999659`。
<a id="s204"></a>
- **S204 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v1_qwen06_pair_seed21_ml2/token_opd/run_card.json`；SHA256 `7cf7028c9904833323bb89c39261c2521a1e635f9b28927d43e30ca16835326e`。
<a id="s205"></a>
- **S205 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v1_qwen06_pair_seed21_ml2/token_opd/acceptance.json`；SHA256 `2348da92b92db655a9d0f372fb830dcb53b240c3deebea1f9ea57f6f454c8677`。
<a id="s206"></a>
- **S206 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v1_qwen06_pair_seed21_ml2/block3_mean/run_card.json`；SHA256 `0ec1c5165f9cc43cb47cbf1e91cb2542885a4b74df578d65bf183f43dec5eacb`。
<a id="s207"></a>
- **S207 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v4_qwen06_nonthinking_token_seed21_ml2/token_opd/run_card.json`；SHA256 `32ea368879999cf3603724f43773dea8146beb20e27306bf2a5a291635b4a6b8`。
<a id="s208"></a>
- **S208 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v4_qwen06_nonthinking_token_seed21_ml2/token_opd/acceptance.json`；SHA256 `c7a6e0aecdfb0356346da4592ac4e72225eae3a2f90b0c5cec3d9458802d8bec`。
<a id="s209"></a>
- **S209 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/block3_mean/run_card.json`；SHA256 `a74c815b643f8ccf8eafd34aabe59386fd2249188651536d545f22d0d48e93fb`。
<a id="s210"></a>
- **S210 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/block3_mean/acceptance.json`；SHA256 `499ff5142a7a03b53df76892a72c9e9da4ef116b3e86f48b6d3295877a970dc9`。
<a id="s211"></a>
- **S211 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v1_llama32_1b_3b_nonthinking_seed21_ml2/token_opd/run_card.json`；SHA256 `856b9d6c725649bed16691299393c16a5e7597c4d224396a1c9e48c11c8f6777`。
<a id="s212"></a>
- **S212 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v1_llama32_1b_3b_nonthinking_seed21_ml2/block3_mean/run_card.json`；SHA256 `e7f9a8ea5ebf47e430bb4a4666225734498f7ca8181b5832c7ef2ba7c6433827`。
<a id="s213"></a>
- **S213 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/token_opd/run_card.json`；SHA256 `9e9da4691e489d27d42d29374484ec113c2b8e2aefcaadeed66f6c6eb35ef195`。
<a id="s214"></a>
- **S214 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/token_opd/acceptance.json`；SHA256 `b5940011156b45f957f7cb3fda4c8c334c7d600c42098ec980ff89cfeb250cd4`。
<a id="s215"></a>
- **S215 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/block3_mean/run_card.json`；SHA256 `9dfa866d6e17571f7103ddad00351bb34608691aa96d6aed10abe412b078a257`。
<a id="s216"></a>
- **S216 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/block3_mean/acceptance.json`；SHA256 `ada5250874ca7802d0d46d71dc6e01058dc882859020d603afa31a69416f6943`。
<a id="s217"></a>
- **S217 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260919v1_qwen4_blockfirst_seed21_ml2/block3_mean/run_card.json`；SHA256 `029823f39e0ed8adff06b9fc05881fd06c8ddc83fc311e64a6c87ce9c45d350c`。
<a id="s218"></a>
- **S218 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260919v1_qwen4_blockfirst_seed21_ml2/token_opd/run_card.json`；SHA256 `1d26c5df8deefaa471b4cb7854a40153de0966efacece6f5be93a52922c4968a`。
<a id="s219"></a>
- **S219 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2/block3_mean/run_card.json`；SHA256 `e89ec002b920cbfdb9076b8d0c1c762b574f0113aa109abb74765029ab2fbe7f`。
<a id="s220"></a>
- **S220 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2/block3_mean/acceptance.json`；SHA256 `38e817f8647d89f8a686e69b9a13c1cad843a7981740ec6dbd4f44b1b478c8ac`。
<a id="s221"></a>
- **S221 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_token_seed21_ml2/token_opd/run_card.json`；SHA256 `3dd3be3ba4046b365f31dd3bc266bc26f4bc9530bdd399390d88c3dd44b4f2d8`。
<a id="s222"></a>
- **S222 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_token_seed21_ml2/token_opd/acceptance.json`；SHA256 `6128ede3a8541077015c88899e376f97ae6f34f7252d45b5a88dd2afe674a167`。
<a id="s223"></a>
- **S223 [C]** [docs/results/2026-09-19-llama-prompt-probe-summary.json](../../../docs/results/2026-09-19-llama-prompt-probe-summary.json)；SHA256 `8d5a9954d25ab9c15d1980b76142768ab904d24fbc4a4f3d61863cb0b4e16d9e`。
<a id="s224"></a>
- **S224 [C]** [docs/results/2026-09-20-qwen-completion-inference-summary.json](../../../docs/results/2026-09-20-qwen-completion-inference-summary.json)；SHA256 `f6efe41a3176a7d3d36a94713f932d846a7d0c84c0335577ec0778a23fb258a4`。
<a id="s225"></a>
- **S225 [C]** [results/2026-09-13-truncation-probe.json](../../../results/2026-09-13-truncation-probe.json)；SHA256 `c5c7d9d79885142d71cdec905454ff70e972fdb93a127028aaaf3c1dd378cc00`。
<a id="s226"></a>
- **S226 [C]** [docs/results/2026-09-20-qwen-completion-gate-summary.json](../../../docs/results/2026-09-20-qwen-completion-gate-summary.json)；SHA256 `e6641410ecddb6eb4193b4265b69028445a624926b6c9f53cc7818c770582d3d`。
<a id="s227"></a>
- **S227 [C]** [results/early_stopping_curve_std_topk.json](../../../results/early_stopping_curve_std_topk.json)；SHA256 `d036a1c7a18b2b0fc5fe59bc402d67a5849eae76b0f0f736929cc35a65782e89`。
<a id="s228"></a>
- **S228 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260911v2r1_sliding_window_seed21_ml2/recoveries/jsonl_20260912_r1/queue_state.json`；SHA256 `31657002822bf0c63de39dda70bcb505909a212de5b43d5b3e8993cdd690ca44`。
<a id="s229"></a>
- **S229 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v2_qwen06_nonthinking_token_seed21_ml2/queue_state.json`；SHA256 `049a7df08c87fba42a5dd5e911aad616c31113702324529bf35cb200b4450012`。
<a id="s230"></a>
- **S230 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v3_qwen06_nonthinking_token_seed21_ml2/queue_state.json`；SHA256 `40247eaa9f72f18ded0320fa443c6c7c0103a1f2e68ef0b56bc21b633a2c5a64`。
<a id="s231"></a>
- **S231 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260913v4_qwen06_nonthinking_token_seed21_ml2/queue_state.json`；SHA256 `7322c532c46ac4b553690c3a367ec722c5e4e029f6b579417db21069b0d5ff15`。
<a id="s232"></a>
- **S232 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/queue_state.json`；SHA256 `19e07419cc7a951e5281ce07e988cfbc764e14f612127bf389011b2db8d79c8f`。
<a id="s233"></a>
- **S233 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2/queue_state.json`；SHA256 `408001d59f11267308ea5e346a886d449f8bf48d9705a8dd852533102ec11275`。
<a id="s234"></a>
- **S234 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v2_historical17_reeval_llama_seed21_ml2/queue_state.json`；SHA256 `b7bd7b40257b1ffa4b3487935290dbd62824ecfe83bef8947f54d7c4009ca5b8`。
<a id="s235"></a>
- **S235 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260918v4_llama32_historical17_recovery_seed21_ml2/queue_state.json`；SHA256 `30b698b8fe933d782513af3d9e2a9c35ecae78de0dd5b248de3d447fe77a3f81`。
<a id="s236"></a>
- **S236 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_gate_seed21_ml2/queue_state.json`；SHA256 `878fa9bd9ea136978a0d17dd1e2de279326f87799be694be203c3426824ebe44`。
<a id="s237"></a>
- **S237 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v2_qwen4_completion_blockfirst_seed21_ml2/queue_state.json`；SHA256 `4b65f07b037ba04d93578964e9ee354e7dbca459dec580be7b6e432c9f137be9`。
<a id="s238"></a>
- **S238 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2/queue_state.json`；SHA256 `bef89aec812c2927dc127ab38709cabe6d6298547eb00f55f87d12f0d427a155`。

<a id="s239"></a>
- **S239 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/README.md`；SHA256 `25889d058465bc1ff4372e91148f67569e4e05d27dd87a76d058f6720e7ff248`。

<a id="s240"></a>
- **S240 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/data/math_opd_dapo17k_hf_full_eval4/manifest.json`；SHA256 `65f8512b46565f0dfb790bdf56be9fe09207a256009d04286d5c51d71eafce4d`。

<a id="s241"></a>
- **S241 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/scripts/prepare_dapo17k_revisiting_math.py`；SHA256 `7e80040627b70ec420378a23417634689ebad18f10143cbff029a1da0d2dd939`。

<a id="s242"></a>
- **S242 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/src/OPD/README.md`；SHA256 `b5fde7f75a820154847e6bd95957e6093f8524d6b858b96de8f94d908c0916a2`。

<a id="s243"></a>
- **S243 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/.gitattributes.metadata`；SHA256 `8aee95cef86a1367b907940c97277f38255f9786f4349ef8edf4aef3d73f28d8`。

<a id="s244"></a>
- **S244 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/README.md.metadata`；SHA256 `5b17ed1c401d5f566c16b81a04c2f66afca38addf8c260fc1ca10e70386cbcd1`。

<a id="s245"></a>
- **S245 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/added_tokens.json.metadata`；SHA256 `04ba5c39374d9abfa5e255defa39707766f6d9ce1a91251c32006d6514ed053f`。

<a id="s246"></a>
- **S246 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/chat_template.jinja.metadata`；SHA256 `80f11a532679a4de9adbb3088fa874cdffc954288d8f648cc5f94dbcd396daa3`。

<a id="s247"></a>
- **S247 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/config.json.metadata`；SHA256 `bde17d7db70e17a48353f441d3947b71128703f5e226c137f8e77c379de2db15`。

<a id="s248"></a>
- **S248 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/generation_config.json.metadata`；SHA256 `d2c25cf7db14c3c345b9d216924a5f1682f4ca37ae8902f2fe39b1ab7f1eb8f0`。

<a id="s249"></a>
- **S249 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/grpo.sh.metadata`；SHA256 `af9cc3a551d973a3a1f1727aa45790e82896be08961c3671b06ea28b1b0131cd`。

<a id="s250"></a>
- **S250 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/merges.txt.metadata`；SHA256 `21eda7ed59d51079acf6cc5bc40eb366b39d2dd4f6638105615fa3852665e7b5`。

<a id="s251"></a>
- **S251 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/model-00001-of-00002.safetensors.metadata`；SHA256 `3dbb75a9efd59eefe5d50d6ff7b9f6e5ddf42aa274971a8f55b26b55ebe12588`。

<a id="s252"></a>
- **S252 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/model-00002-of-00002.safetensors.metadata`；SHA256 `8bf704454a5ac5eebdc20d4e3ac6c25154b5f38da38d518243d4f8d71da04dd4`。

<a id="s253"></a>
- **S253 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/model.safetensors.index.json.metadata`；SHA256 `a582283cf00b3f56daa8a4ac3ea489b16db2ff9b965fc7109e8f0c5927f4c51e`。

<a id="s254"></a>
- **S254 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/special_tokens_map.json.metadata`；SHA256 `a0deb3ec2a917c8b4a98611b59779dd94608368cd2c7f8eb5070e3859dd77b16`。

<a id="s255"></a>
- **S255 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/tokenizer.json.metadata`；SHA256 `bf3e265975885e669fbd642d207b285fe7c8bf21f7162fa52f42c3c72c03b1d8`。

<a id="s256"></a>
- **S256 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/tokenizer_config.json.metadata`；SHA256 `07f74a3393640a9d1800bac0ec1019312c1ff92e792a5cea3e5a78e4b552f63e`。

<a id="s257"></a>
- **S257 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/models/Qwen3-4B-Base-GRPO/.cache/huggingface/download/vocab.json.metadata`；SHA256 `40384127bcaf401f2a782443f562a6bc8a715ba3cf59dfcddf8ea8fcf0d27024`。

<a id="s258"></a>
- **S258 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/src/OPD/datasets/test_data/AIME25/test.parquet`；SHA256 `14ee9a0e8d79cba76c62247bc5ebee1647e023eccf60e31994e279113777dad2`。

<a id="s259"></a>
- **S259 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/src/OPD/datasets/test_data/AMC23/test.parquet`；SHA256 `0f933978194b73fc091bd667af7d9056d3d745c9d9e3577b5adcad77f8f1ae7c`。

<a id="s260"></a>
- **S260 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/data/math_opd_dapo17k_hf_full_eval4/eval_jsonl/aime24.jsonl`；SHA256 `6aee5ed24b811ebc210de62f4606600744bd7e90ed143d835781c1a8783f8fa8`。

<a id="s261"></a>
- **S261 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/data/math_opd_dapo17k_hf_full_eval4/eval_jsonl/math500.jsonl`；SHA256 `cc164b47b60771eeda257bcb9a2068940cc0706bb8bb79b6635af4c1d4534501`。

<a id="s262"></a>
- **S262 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/data/math_opd_dapo17k_hf_full_eval4/eval_jsonl/aime25.jsonl`；SHA256 `daedf1e406b9d73ca25e9696c92614e300e34c4d09376dacaa465b01bb398d43`。

<a id="s263"></a>
- **S263 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/data/math_opd_dapo17k_hf_full_eval4/eval_jsonl/amc23.jsonl`；SHA256 `a07c52c0098a536a2d03e01e531b02c5e9d0a1f4e66d04a0793c0345e1730b45`。

<a id="s264"></a>
- **S264 [A]** `ml2:/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/data/math_opd_dapo17k_hf_full_eval4/train.parquet`；SHA256 `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`（历史manifest登记，未重算整文件；本次只读完整投影列）。
