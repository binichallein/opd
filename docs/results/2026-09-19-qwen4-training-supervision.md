# Qwen4 Block3 训练监督

后续更新：用户于21:32要求停止，正式训练停在Step4。初始循环诊断已经完成，
见[停止与根因排查](2026-09-19-qwen4-initial-loop-incident.md)。下文保留当时监督快照，
不是最新运行状态；不得据此自动恢复原队列。

## 范围

2026-09-19 用户要求监督训练。仅访问 ml2，不访问 train；不修改冻结 runtime、
算法、模型、提示、数据、seed 或保存策略。最新状态必须读取服务器，本文是快照。

- Run：`runs/20260919v1_qwen4_blockfirst_seed21_ml2`。
- ROOT：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
- 冻结 runtime：`57ae7dfec10e206dd441a4c70571308c09f64a03`。
- Controller PID1047903；Block3 probe1 PID1049174；probe2 PID1066486。
- 合同：[Block3-first 计划](../plans/2026-09-19-qwen4-blockfirst.md)。
- 以下时间均为北京时间；这是启动/恢复监督，不是 benchmark 评测或方法有效性结论。

## Probe1：已完成更新与保存

- 20:41 已进入真实 vLLM rollout；20:46 第一批 raw archive 已生成。
- 20:47 已完成第一步更新，随后写入约47 GiB完整 checkpoint。
- 20:50:19 队列进入 probe2，说明 probe1 退出及 checkpoint 验收已通过。
- 文件：`probes/block3_mean/rollouts/probe1/step_000001/raw.jsonl.gz`。
- SHA256：`9e05fce70fb81eaf9063f3868e1d36e11e2c2febefbcb321625ae5da6ea4cd81`。
- 手动调用冻结代码的 `audit_rollouts(..., [1])` 通过：32条、32个唯一轨迹ID，
  实际 prompt IDs、采样参数、EOS151643、完整训练mask及log-prob符合合同。
- log-prob、熵、advantage、post-update block ratio的非有限计数均为0；未观察到OOM。

| Probe1 指标 | 数值 |
|---|---:|
| 裁剪前 grad norm | 4.984056 |
| PG loss | 0.025519 |
| Student entropy | 0.052153 |
| Teacher entropy | 0.044062 |
| Top-16 overlap ratio | 0.917605 |
| SignFlipRate | 0.037388 |
| WeightedSignFlipRate | 0.080399 |
| NormalizedLeakage | 1.019149 |
| 平均 response 长度 | 8540.25 |
| 实际 length stop | 16/32 |
| 生成 think 标签 | 0/32 |

### 初始输出质量风险

这32条来自4道题，每题8条。实际只有4种不同输出，符合历史固定请求seed造成的
同题重复限制，不能把它们当成32个独立样本，也不能把本批50%截断外推到全训练集。

四道题的输出长度分别为844、16384、16384、549；两个长输出存在明显重复循环，
另有输出开头出现多语言杂乱文本。两条较短输出可继续作数学解答，但人工阅读发现
解题/答案错误。本次没有对这些训练样本进行完整历史grader计分。

**这些输出采样于第一次参数更新之前，不能作为Block3导致collapse的证据。**
低平均熵也不代表推理质量好：长重复序列可以主导按token聚合的均值。
当前输入/存档审计没有发现协议偏差。下述CPU局部复算也不能替代完整GPU一致性测试，
因此不把根因定性为某一个组件。保留现场并继续观察，不静默改提示或算法。

### CPU局部概率复核

20:54-20:57，在独立进程用原始4B-Base和原环境的Transformers做teacher forcing。
`CUDA_VISIBLE_DEVICES`为空，BF16、CPU SDPA、4线程、eval/inference mode，
不生成新轨迹、不改随机seed、不占训练GPU，也不是提前进行初始benchmark评测。

对每道题使用存档中的原始prompt IDs及前4个response IDs，取对应的未截断词表
log-softmax，与vLLM原始log-prob比较。32条存档的response_text均与原始tokenizer
对response IDs的解码完全一致。四道题的结果如下：

| 数据行index | 前4token最大绝对log-prob差 | 首token之前的更高概率质量 |
|---|---:|---:|
| 1434553 | 0.153991 | 0.838172 |
| 1742959 | 0.048422 | 0.847754 |
| 69824 | 0.107134 | 0.530892 |
| 1225498 | 0.043683 | 0.000000 |

所检查16个token的更高概率质量均小于0.9，没有看到采样出CPU复算nucleus以外token
的反例。两个杂乱开头首token本身log-prob约为-10.8/-10.9，虽然概率低，仍在很宽的
top-p支持集中。不能简单以“采到低概率token”认定top-p配置失效。

这些结果仅说明局部概率没有数量级错位，并非CPU与GPU逐位一致，也不能证明完整
sampling实现或所有长序列都正确。没有据此排除所有推理后端问题或证明质量风险根因。

21:01额外核对历史1.7B tokenizer：两个Base模型的chat_template文本相同，旧tokenizer
EOS同为151643；用旧tokenizer和历史Math指令重建32条prompt，均与新实验实际输入
token IDs完全一致。模板为原生Qwen user/assistant边界，think要求位于user正文，
assistant前缀没有新增强制think标签。

### 实际 checkpoint 内容核验

20:51 使用CPU和mmap逐个读取四rank的optimizer及extra_state，并读取`data.pt`：

- 四rank optimizer state中所有step均为1。
- 四rank scheduler `last_epoch=1`，学习率均为`2e-6`。
- 四rank均含CPU、NumPy、Python random、CUDA RNG状态。
- dataloader `_snapshot_step=1`，sampler `samples_yielded=4`。
- 模型、optimizer、extra_state文件都存在；不是仅有HF权重的不可恢复保存。

## 恢复与正式训练

20:51:27快照：probe2仍在初始化；尚未观察到加载完成、Step2更新或正式训练开始。
必须等待实际四rank加载日志、Step2状态推进、轨迹审计和磁盘预算检查。
正式训练仍须从原始4B-Base独立初始化，不得使用探针权重。

20:58:46快照：已观察到`Resuming from .../global_step_1`及四rank各自的model、
optim、extra_state加载日志。进度从1/2继续，正在生成Step2轨迹；尚未保存Step2。

21:04:30已完成恢复后的Step2更新。grad norm1.192241、PG loss0.007556、学生熵
0.023671、教师熵0.017632、Top-16 overlap0.899296；非有限计数仍为0。
第二批32条raw SHA为`dd6765e06c054b31015a49af7334a38316df244027460ddaed03e9545789f0c9`，
24条length stop，8条长度1712自然结束，4种唯一输出。三个长输出有明显重复/偏题。
两步题目不同，不能用50%到75%的单批截断变化建立训练恶化的因果结论。

21:07:21已取得全部门禁通过证据：`resume_gate.json`、`rollout_acceptance.json`、
`storage_gate.json`均passed；64条raw覆盖完整，两个位置快照已保存。实际每份完整
checkpoint49,842,447,829字节，按剩余保留/合并计划计算需765,980,417,863字节，
挂载点报告的剩余空间满足检查；这不是对外部对象存储配额的额外保证。

21:09 CPU实际读取Step2四rankoptimizer/extra_state和dataloader：optimizer所有step
均为2，scheduler均为2，LR2e-6，四类RNG齐全，sampler消费8道题。
验证了训练状态恢复与推进，不声称vLLM恢复前后逐位一致。

### 退出清理告警

probe2在`Training Progress: 2/2`、Step2指标和最终日志之后，
`MathMultiProcessEnv.__del__`中出现ignored exception：DataLoader worker1075104
被Killed。探针及其验收的exit_code均为0，模型/优化器/数据状态和轨迹全部验收通过。
已完成的Llama v4 Token/Block3正式日志末尾也存在同类退出清理提示。

据此记录为未造成该探针失败的退出阶段告警，而非误报成训练途中CUDA OOM。
没有取得该SIGKILL的内核级来源证据，不声称已经证明具体成因；不屏蔽日志、
不在冻结runtime中临时修改析构行为。若正式训练过程中出现worker死亡，须另行处理。

### 正式任务已启动

- 21:07:37正式Block3进程PID1079900启动。
- `block3_mean/logs/nohup.log`，原始4B-Base初始化、resume关闭。
- 保存/评测50、100、150、200，初始评测和Token任务仍在其后。
- 21:08快照仍为初始化；尚未产生正式Step1指标，不能把探针的两步计入正式200步。

### 正式Step1已完成

21:10实际命令与冻结artifact再核对通过：原始学生、resume disable、200步及
50/100/150/200保存列表均存在于实际启动命令，两组6个冻结配置/命令/清单哈希未变。

21:21:31确认正式Step1已完成，进度1/200，正在第2批生成。记录值如下：

- 裁剪前grad norm4.982413，PG loss0.025519，所有标量数值/非有限计数检查通过。
- 学生熵0.052153、教师熵0.044062、Top-16 overlap0.917605。
- 首批16/32实际length stop，平均长度8540.25；初始质量风险与probe1一致。
- 首批raw SHA：`ee59989a97620a74c687368df35c8ff452721fd8b0f65b2c36380c5cd6bfeace`。
- 冻结`audit_rollouts(formal, [1])`通过；与probe1逐条对比，prompt顺序/IDs、
  response IDs、结束原因、sampling、EOS和完整训练mask均一致。
- training rollout log-prob最大差`2.1457672119140625e-05`；梯度值也有微小差异，
  不声称GPU计算逐位确定。原始初始化与首批一致性核验没有发现误用探针权重的证据。
- 正式Step1没有checkpoint是预期行为：完整保存从Step50开始，Step1原始轨迹和
  诊断已保存。不能为方便监督临时增加/替换保存策略。

本次监督未修改任何训练算法、配置或冻结runtime。尚无本师生对的benchmark结果，
也没有证明Block3有效或无效；后续仍按Block3四权重评测、初始评测、Token训练/评测执行。
