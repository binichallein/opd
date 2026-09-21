# 数据集字符串重叠审计

审计完成：2026-09-20T18:30:12.099946+00:00（北京时间2026-09-21 02:30:12）。只读ml2 CPU，无GPU/模型加载、训练或评测启动，无数据过滤。仅本文件和同目录 `dataset_overlap_audit.json` 为本任务写入文件。

## 结果

单位为**held-out题目记录的整数数量**，不是百分比、响应数或训练次数。

| Benchmark | held-out题数 | exact重叠 | 空白折叠+casefold重叠 | 规范化后新增 |
|---|---:|---:|---:|---:|
| MATH500 | 500 | 2 | 4 | 2 |
| AIME24 | 30 | 0 | 0 | 0 |
| AIME25 | 30 | 0 | 0 | 0 |
| AMC23 | 83 | 0 | 0 | 0 |

各评测集在两种比较规则下均无自身重复比较键，因此此表的匹配记录数也等于各自唯一匹配键数。训练周期内精确唯一题面17,237个；空白折叠+casefold后唯一键17,186个。

## 比较定义与范围

- **Exact：** 训练 `env_kwargs.question` 与held-out JSONL的 `problem` 按Python Unicode字符串逐字相等；不strip、不去指令、不改大小写、不比较答案。使用原始题面字段，不比较带ChatML/completion包装的prompt。
- **Whitespace + casefold：** 两侧均执行 `" ".join(question.split()).casefold()`。Unicode空白折为单个ASCII空格并去掉首尾空白，再casefold；没有NFC/NFKC、数学等价、标点或LaTeX改写。Python 3.12.13，Unicode数据库 15.0.0。
- 本次只读训练池前17,917行的question/index列。此前完整1,791,700行扫描已证明题答序列按该周期重复100次，本次依此覆盖训练池的字符串集合；**没有重新扫描100段，也没有重新计算1.6GB训练文件SHA**。
- 既有完整扫描证据：[historical_evidence.json](historical_evidence.json)，字段 `data_provenance.dapo_train.observed_pool`，扫描时间 `2026-09-20T17:14:32.722302+00:00`；本次读取该审计文件的SHA为 `fbdf313b0e2b1d196e949a705ca43d7f452b8dc39cc5a4468f4e5cb1548a2b71`。
- 当前训练文件仍为1,791,700行、1,606,646,755字节；前周期index为0至17,916，精确unique仍为17,237，读取前后stat一致。这些检查不是整个训练文件本次重新hash的替代品。
- 四份JSONL共643题，逐题按source/index核对 `problem == test.parquet.env_kwargs.question`；JSONL与held-out Parquet的全文件SHA均与此前记录一致。
- 字典集合匹配结果另用逐项线性membership独立核对，两种规则、全部643题均一致。未传回任何题面原文。

## 匹配来源ID

训练index为前周期的0-based物理行号，亦与该周期 `extra_info.index` 相同。完整池相应位次可依既有周期证明写为 `index + r*17917`（r=0..99），不表示训练实际使用过这些位次。

| held-out ID | 匹配类型 | 训练周期index | held-out题面UTF-8 SHA256 |
|---|---|---:|---|
| `math500/test/80` | exact及规范化均匹配 | 6216 | `07e22408c9d86bf198748e6e0ffd321d2d6b8d605ebab6f9a119cca4c7ce5785` |
| `math500/test/276` | exact及规范化均匹配 | 17762 | `a153bf646bb8dfb046d7c75433a9e56cd7c552526ee5ec4085c5b284d4b6813a` |
| `math500/test/285` | 仅空白折叠+casefold匹配 | 6170 | `d03a1b8fc03cf76a4f4ec3147a0a64c4adb66c431a993cb3ca0b4515d9fdec7b` |
| `math500/test/412` | 仅空白折叠+casefold匹配 | 6090 | `2b8fd0ef03dada66be92d8bc85a3fd19557ceaa619ab7d0934da75063319b54b` |

JSON保留每个匹配的训练题面SHA、规范化比较键SHA及ID列表SHA；不包含题面原文。MATH500 exact对应训练周期内2行，规范化对应4行；完整池中200/400个出现位次仅为100倍周期推导值。

## 文件与列哈希

| 对象 | SHA256 |
|---|---|
| train.parquet（**历史manifest登记，本次未重算**） | `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf` |
| 前17,917题面有序列表（本次计算） | `521eb6a681d0d019b8566c3045a6424635c31f802caaf226290b3aa4a6ade2b1` |
| 前17,917题面规范化有序列表（本次计算） | `4bc079838bd760104db20eea21370fa4a00952366d72cd5bc3a22e9b63df208f` |
| test.parquet（本次全文件SHA） | `c5c9f1b3f65eb40053cae14c005c39e29443d04acbbabba873e98d64818a9680` |
| MATH500 eval JSONL（本次全文件SHA） | `cc164b47b60771eeda257bcb9a2068940cc0706bb8bb79b6635af4c1d4534501` |
| AIME24 eval JSONL（本次全文件SHA） | `6aee5ed24b811ebc210de62f4606600744bd7e90ed143d835781c1a8783f8fa8` |
| AIME25 eval JSONL（本次全文件SHA） | `daedf1e406b9d73ca25e9696c92614e300e34c4d09376dacaa465b01bb398d43` |
| AMC23 eval JSONL（本次全文件SHA） | `a07c52c0098a536a2d03e01e531b02c5e9d0a1f4e66d04a0793c0345e1730b45` |
| 历史数据manifest.json（本次全文件SHA） | `65f8512b46565f0dfb790bdf56be9fe09207a256009d04286d5c51d71eafce4d` |

列表哈希使用 `json.dumps(list, ensure_ascii=False, separators=(",", ":")).encode("utf-8")`；有序列表保持文件顺序，unique列表先按Python Unicode字符串排序。每个题面/比较键哈希直接作用于其UTF-8字节。各benchmark有序题面列表及训练unique列表哈希另见JSON。

## 可复核边界

- **不能声称语义去污染。** Exact的2个MATH500匹配说明该评测集并非与此训练池严格字符串互斥；规范化新增2个匹配使用不同规则，不能回填为exact。AIME24/AIME25/AMC23的零值不排除改写、等价数学题或其他形式重叠。
- 未检查200updates实际抽到的800个prompt位次是否命中这4题，未审计师生预训练池；不据此推断记忆、分数膨胀或因果影响。
- 未修改训练/评测集合或任何模型分数；当前正在进行的评测不受本审计改变。
- 完整只读审计代码见JSON的 `audit_program.python`，源码SHA为 `27afc76a076c7ebd4300c6b9040d28b2d4d883493cbde4c3276ca4ef980957f1`。程序仅用现有Python/pyarrow及标准库，线程上限1；大整数stat字段使用十进制字符串保存，避免JSON消费者精度丢失。
