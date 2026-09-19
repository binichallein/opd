# Llama 原生聊天模板与训练后退化排查

## 范围与状态

本文回应“旧 Llama 是否也遇到 Qwen Base 的聊天格式不适配”。全部256条GPU输出已完成，
完整性/配对/终止语义/历史grader核验通过。教师最后一组于23:05北京时间完成，
随后两个进程退出，四张GPU恢复空闲。没有提前截短或漏掉未结束的长轨迹。

- 只访问 ml2。未恢复 Qwen4 v1，未启动或修改 Llama 训练。
- 原始学生为 ModelScope `LLM-Research/Llama-3.2-1B-Instruct`，revision
  `d3e551343d4d81508a0d226656b826c217e463cd`。
- 原始教师为 `LLM-Research/Llama-3.2-3B-Instruct`，revision
  `4e7231b81c151c73632184994ac9a0149fcb22fd`。
- 两者都是 Instruct，不是 Base；权重/配置/分词器按既有 ModelScope 清单核验。
- 新代码提交 `9cf0f42`；运行副本位于独立诊断目录，旧冻结部署未改。
- 远端根目录：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
- 新诊断目录：`diagnostics/20260919_llama_prompt_check`。启动 PID1117184/1117185，
  GPU0/1；实际状态由进程和最终summary判断，不能把此PID记录当仍在运行的证据。

## 格式审计

旧训练实际输入是一个原生BOS、system/user/assistant header及原生EOT；没有Qwen
`<|im_start|>`、`<|im_end|>`。system模板日期固定为18 Sep 2026，没有随本次日期变化。
原生停止ID为128001、128008、128009，ignore_eos=False。
这种聊天结构与[Meta的Llama3.2官方提示格式](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md)
一致；官方另为Base模型列出completion输入，不能将两类模型混用。

“去think”只删除用户消息中的`conduct reasoning inside <think> and </think> and `。
它不是关闭Llama的某个原生thinking开关。`enable_thinking`不是该模板的模式控制机制。

## GPU 对照设计

复用旧Token正式训练Step1-4中的16道题，每题seed21/22两次；两种原始模型各四组。
只使用这些归档中的问题和输入，不把Step2以后的已更新模型回答当作原始模型回答。
每组32条、总计256条。BF16、vLLM0.11.0、TP1、temperature1、top-p0.9、top-k-1、
max tokens16384、max model len18432、memory0.6、eager、prefix caching开启。

1. `historical`：原生chat加历史think要求，实际输入ID逐条与归档一致。
2. `no_think`：只删除上述think短语，保留原生模板及其余用户内容。
3. `completion`：与第二组相同的用户内容，改为BOS+文本+Solution，不使用chat。
4. `eval`：实际历史评测用的原生chat提示，简洁的step-by-step/boxed指令，不要求think。

第三组移除system/header并增加Solution，因此是整体提示格式干预，不是单个边界token消融。
这是固定训练题诊断，不是完整benchmark，不据此声称统计显著或泛化提升。

## 完整结果

以下每格分母均为32，表示输出条数，不是32道独立题。

| 提示 | 学生截断 | 教师截断 | 学生高重复尾部 | 教师高重复尾部 |
|---|---:|---:|---:|---:|
| 历史原生chat+think要求 | 1/32 | 2/32 | 1/32 | 2/32 |
| 只删除think要求 | 1/32 | 4/32 | 0/32 | 3/32 |
| 去chat，改completion | 4/32 | 4/32 | 2/32 | 2/32 |
| 实际评测原生chat提示 | 1/32 | 0/32 | 1/32 | 0/32 |

“高重复尾部”沿用末2048 token的重复4-gram比例>=0.9；严格周期检测另存于JSON。
两者都是启发式，不会检测全部语义重复，也不表示没有重复的回答一定正确。

| 提示 | 学生判对 | 教师判对 | 学生历史格式错误 | 教师历史格式错误 |
|---|---:|---:|---:|---:|
| 历史原生chat+think要求 | 0/32 | 3/32 | 10/32 | 6/32 |
| 只删除think要求 | 1/32 | 2/32 | 16/32 | 9/32 |
| 去chat，改completion | 1/32 | 1/32 | 3/32 | 2/32 |
| 实际评测原生chat提示 | 0/32 | 2/32 | 11/32 | 6/32 |

判题使用既定SHA04f7历史grader，解码时去除原生特殊token，与评测的可见输出口径一致。
256次判题无异常/超时；额外用16个gold构造boxed答案自检，16/16通过。
这里的历史格式错误严格沿用“输出不含`\\boxed`”，不等于数学表达式解析错误。
completion虽然更容易出现boxed，但答题结果没有相应改善，是格式与正确性分离的例子。
不能从这16道题推断两模型在全部DAPO训练池或benchmark上的准确率。

人工复核例：整数求和题index1742959的正确答案是287；教师在历史提示下给出208、
-1761.5，在评测提示下给出243、-491。多数不是纯粹的答案抽取失败，推导也确有错误。
“判对”只衡量最终答案，不证明整条推理正确。

### 如何解释

1. **不支持“Llama也是Base套ChatML”的解释。** 实际模型是Instruct，使用正确原生模板；
   此次去chat没有修复截断，学生反而从1/32增至4/32。不能照搬Qwen Base的处理。
2. **提示内容仍会影响行为。** 去掉think要求后两模型均0/32生成think标签，但不带来
   一致的截断改善。完整评测提示对教师的截断更好；这是小样本观察，不证明该提示普适最优。
3. **两边都有请求seed复用问题。** 旧训练同题8次重复；本次每题seed21/22，各组均32个
   不同的题目-回答组合。修复的是请求间随机性管理，不是说全局seed21这个数有问题。
4. **Llama还存在训练后的退化和指导能力不足风险。** 原始模型偶尔数学循环，与Block3
   Step28起的高熵无关输出不是同一种现象。两者可能交互，但这次没有通过受控重训确定
   高熵退化的具体原因，也没有证明改prompt或seed就能救回Block3。

下一轮Base按用户批准不用ChatML、统一训推提示、独立派生请求seed、先验收再训练；
Llama Instruct保留原生chat。若重做Llama，建议另立新实验统一训推指令并修正请求seed，
先验证教师在实际训练题/学生前缀上的指导能力，再查block更新的尺度、ratio和信用分配。
这里没有修改旧Llama实验或启动新训练；也不把“独立生成答对更少”直接等同于局部蒸馏无用。

## 已复核的历史证据

- 旧Token与Block3首批32条输出逐条相同，均0/32截断；四题各8个重复副本，只有4条
  不同回答。首批没有周期性尾部循环，但存在数学错误，不能称全部健康或正确。
- 抽查两臂Step1/20/25/28/30/35/50/100/200，共18批576条，归档SHA匹配。
  未发现Qwen ChatML、重复BOS或已经生成原生终止token后仍继续输出的情况。
- Token也有低熵循环，例如Step20为16/32截断、熵0.182；不是完全健康的对照。
- Block3至少Step28已有无关词语串；Step35学生熵9.237，Token同一步为0.447。
  Step50 Block3熵9.626，原始训练首步是0.512。熵是在各自on-policy轨迹上统计，
  不是相同前缀比较，不能据此单独证明某个loss细节的因果作用。
- 旧训练每个请求重复使用seed21会制造重复副本。这是两家模型共享的采样问题，
  但不能单凭它解释为何Block3产生高熵退化而Token未出现同等程度的退化。
- 历史Block3同时改变advantage、joint PPO ratio与归一化；不能将其失败完全归因于
  advantage共享，也不能为改善结果静默除以3而声称算法未变。

原始轨迹与各step证据保存在新目录`historical_audit.json`；历史完整评测仍在原run中。
本地可追踪副本见[历史核验JSON](2026-09-19-llama-historical-prompt-audit.json)。
完整实现分析见[此前诊断](2026-09-19-llama-block3-quality-diagnosis.md)。

## 留存与核验

新目录中各模型/各组都有`inputs.json`、`raw.jsonl`、`summary.json`。raw包括完整文本、
token IDs、逐token logprob、请求seed、真实finish reason。manifest记录模型/代码/源归档哈希。
`audit_llama_prompt_probe.py`已检查完整性、两模型输入/采样配对、终止语义、数值有限性，
并用SHA04f7的历史grader独立判题。结果见[完整诊断JSON](2026-09-19-llama-prompt-probe-summary.json)。
8组均32条，16个不同题目，每题2条；两个模型的同组实际输入和采样配置一致。
环境为PyTorch2.8.0、Transformers4.57.6、vLLM0.11.0。
两个进程退出时均有NCCL未显式destroy的清理warning，日志保留；完整输出/核验通过且
显存已释放，未观察到推理中OOM或异常退出，不将清理warning认定为循环原因。
本地相关测试72 passed；GPU结果另经实际核验，不把单元测试代替实验验收。
