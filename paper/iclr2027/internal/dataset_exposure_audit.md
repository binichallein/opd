# 当前 4B 配对运行的实际训练输入重叠暴露审计

审计完成：2026-09-20T18:42:45.287606+00:00（北京时间 2026-09-21 02:42:45）。
范围：当前 Qwen3-4B completion 的 Block3 / Token 两组，seed21，各 200 updates、4 prompts/update、8 responses/prompt。
仅只读 ml2 CPU；未连接 train，未启动 GPU，未改训练/评测数据、代码或父线程稿件。

## 结论与实际整数

**两组均实际遇到一道 normalized-only MATH500 题：`math500/test/412`，step71 一次 prompt occurrence、8 条 trajectory records。两道 exact 池重叠题未实际出现。**

下表计数均为每组，Block3 与 Token 相同；两组各完整覆盖 **800 次 prompt occurrence / 6,400 条 records**。normalized 包含 exact。

| 前次池审计题目 ID | 池匹配类型 | Block3 exact 次数 | Block3 normalized 次数 | Token exact 次数 | Token normalized 次数 | 命中 step / 每组 records |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| math500/test/80 | exact | 0 | 0 | 0 | 0 | 无 / 0 |
| math500/test/276 | exact | 0 | 0 | 0 | 0 | 无 / 0 |
| math500/test/285 | normalized-only | 0 | 0 | 0 | 0 | 无 / 0 |
| math500/test/412 | normalized-only | 0 | 1 | 0 | 1 | 71 / 8 |

另外直接对四个 heldout JSONL 全部 643 个 `problem` 字符串比较，结果如下。不是只搜索已知四个 source indices。

| Benchmark | heldout 题数 | 每组 exact prompt 次数 | 每组 normalized prompt 次数 | 每组 normalized-only prompt 次数 | 每组 exact / normalized records |
| --- | ---: | ---: | ---: | ---: | --- |
| MATH500 | 500 | 0 | 1 | 1 | 0 / 8 |
| AIME24 | 30 | 0 | 0 | 0 | 0 / 0 |
| AIME25 | 30 | 0 | 0 | 0 | 0 / 0 |
| AMC23 | 83 | 0 | 0 | 0 | 0 / 0 |

两组各有 800 个不同物理 source indices，但只有 **783 个 exact 不同问题 / 782 个 normalized 不同问题**。所以 800 是训练抽取次数，不能写成 800 个 unique questions。同一源题对应的两组采样不能算成两个不同 heldout 题。

## 唯一实际暴露的来源

- 实际记录：step71，第 3 个 prompt（zero-based ordinal 2），archive sample indices 16–23。
- `source_extra_info.index = request_identity.source_index = 1564869`；`1564869 % 17917 = 6090`。
- 原始训练 question SHA256：`06d5aeabb535b6a990b29e1f5210e010d81f9f767a3bbe9e9b07c62c1ca02c08`。
- heldout /412 原始 question SHA256：`2b8fd0ef03dada66be92d8bc85a3fd19557ceaa619ab7d0934da75063319b54b`。两者不同，**不是 exact match**。
- 共同 normalized question SHA256：`9bc2bb593c07b21841848ca78b891b595376d57ca627b41e2bd3c6396fcfb6ff`。
- Block3 step71 `raw.jsonl.gz` SHA256：`3659dfd427b2323998295413d61ce72dc21c04d46b01f5c55bc144e6cf92c42e`。
- Token step71 `raw.jsonl.gz` SHA256：`a4d6f5e6f027e828bccdd2a35f0cffef89a9c322b8b3fddf7c54a4db6ef73f29`。

归档路径为以下 run 根目录加 `rollouts/formal/step_000071/raw.jsonl.gz`：

- Block3：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2/block3_mean`
- Token：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_token_seed21_ml2/token_opd`

## 方法、完整性与配对复核

1. 先查 compact 元数据：两组 `summaries/` 为空；formal 每步仅有压缩 raw archive 及 SHA sidecar。既有 `paired_rollout_acceptance.json` 有 200 步、每步 32 条及两组 archive hashes，但没有题目身份，不能单凭它回答暴露问题。
2. 只读流式解析两组全部 400 个归档。每条重新计算 `source_extra_info.question` 的原始 UTF-8 SHA256，核对 `request_identity.question_sha256` 与 source index，并核对该 question 原文确实在记录的 `prompt_text` 中。**不用 RNG、seed 或 sampler 重建输入顺序。**
3. 每步用实际 source index 分组，验证恰好 4 组且每组 8 records；总计每组 800 次 / 6,400 条。400 个压缩文件的完整 SHA256 均同时符合 sidecar 和已留存 pairing 报告；逐文件读取前后 size/mtime/inode 不变。
4. exact 为原始字符串相等；normalized **仅** `" ".join(question.split()).casefold()`，不做 Unicode NFC/NFKC、LaTeX、标点或语义处理。四个小型 heldout JSONL 重新计算文件 SHA，与前次池审计一致；在远端内存直接比较字符串，既核对四道候选身份，也计算四 benchmark 全量匹配次数。
5. 对两组完整 6,400 条逐序比较 request identity、source index、question 原始/normalized hash、prompt text hash、prompt token IDs hash、sampling settings（包括 request seed）、step/sample index/request turn、protocol、source commit 与 enable_thinking。全部相同；不要求生成 response 相同。
6. 完整 ordered input metadata SHA256 两组均为 `d749989533243a0147ccb857208993bdce876e307c25a9bba42225a9e21265c0`；800 次 ordered prompt metadata SHA256 均为 `1713aa7552ba7fc57088046073c8e3b1ac2b1a43742ef64d901f4d307bcc93c2`。canonicalization 及全部逐步哈希见 JSON。

## Checkpoint 前缀

仅报告已记录训练输入的累计暴露，不表示评测完成或得分变化。两组相同：

| checkpoint step | 累计 prompt 次数 | exact 匹配次数 | normalized-only 匹配次数 | 匹配 records |
| --- | ---: | ---: | ---: | ---: |
| 50 | 200 | 0 | 0 | 0 |
| 100 | 400 | 0 | 1 | 8 |
| 150 | 600 | 0 | 1 | 8 |
| 200 | 800 | 0 | 1 | 8 |

## 有界执行与证据文件

正式完整扫描：单进程 CPU、nice15、88.221 秒；压缩读取 **406,560,948 bytes**，远端解压解析 **4,794,974,588 bytes**。限制为 120 秒、500 MiB compressed、8 GiB decompressed、32 MiB/line，均未触发。正式扫描前另有少量 step1 schema/计时读取，不包含在这两个正式 totals 中。归档包含 responses，解析后仅保留 prompt 身份；没有 question/response 原文或 token 数组外传，也未写远端文件。仅导出约 510 KB 结构化哈希/计数元数据。

本次 Python 3.13.2 / Unicode 15.1.0；前次池审计 Python 3.12.13 / Unicode 15.0.0。四道候选使用已有固定 question/key hashes 交叉核对，同时按本次同一 normalization 直接比较 heldout 全量字符串；运行时版本均明确记录。

- 前次池重叠审计：`paper/iclr2027/internal/dataset_overlap_audit.json`，SHA256 `31bf1b79b25aec56d7ab055cd192f0f649709a8845e71171e751e6c95996ed1b`。
- 既有 `paired_rollout_acceptance.json`（Token run 的上一级目录），本次重新计算 SHA256 `7f692d3d47607d79df1bf8cfd6e06c6276e8f007684eeaaf46edc0b9d45870ed`。
- 训练 data manifest 两组 SHA256 同为 `65f8512b46565f0dfb790bdf56be9fe09207a256009d04286d5c51d71eafce4d`；训练文件完整哈希来自 manifest，没有重新读取全训练池。
- 配套 `dataset_exposure_audit.json` 含 11 个小型来源文件的本次哈希、全部 400 个 raw archive 和 sidecar 哈希、800 次共享 prompt metadata、两组分题/分 benchmark 整数、checkpoint 前缀及完整可复核只读程序。
- 审计程序 UTF-8 SHA256：`f97beaa5b268472af0e8dd9d2b7ee5c30c84db8573c13e878162720cd8751af1`。

## 解释边界

应同时保留“训练池有 2 exact + 2 normalized-only heldout 题”和“当前两组各实际遇到 0 exact + 1 normalized-only prompt occurrence”的披露，不能用后者抹去前者。两组输入匹配只说明本次暴露相同，**不证明无污染或已完成 semantic decontamination**。

本次不评估语义近似、答案泄漏、记忆或分数增益；不证明学生预训练/教师 GRPO 数据无重叠；不外推其他历史系列。step71 时点也不能用于性能归因。未筛除数据、重跑评测或修改稿件。

可用于稿件的严格英文表述：
> Across the retained 200-update paired runs, each arm contained zero exact-string heldout matches and one whitespace/casefold-normalized-only match among 800 prompt occurrences (eight of 6,400 rollout records). The same MATH500 item occurred at step 71 in both arms. This string-level exposure audit does not establish semantic decontamination.
