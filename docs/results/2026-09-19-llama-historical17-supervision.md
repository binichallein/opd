# Llama历史提示对照监督记录

## 运行身份

- 用户要求持续监督训练，遇到故障定位修复；仅访问ml2。
- 当前队列：`20260918v4_llama32_historical17_recovery_seed21_ml2`。
- 控制器PID623151，控制器发布`b00ad93385ef367cc88c499d3c89d9feaa515327`。
- 训练和评测固定调用`94be7ea1d659309256c8356681925bb9710895c4`，不随文档提交变化。
- Token和Block3的paired preflight已通过。配置差异仅为算法变体、block大小/
  advantage聚合方式、实验名与各自输出目录。micro-batch=1、vLLM=0.6、
  seed21、200步和原始ModelScope师生保持不变。
- 正式训练从原始学生重新初始化，不从探针或旧的Step146尝试续训。

## Token断点恢复测试

### Step1

探针PID626008于9月18日23:51:31启动，9月19日00:01:17正常退出。
checkpoint审计也正常退出，`passed=true`、无issues/warnings。

| 指标 | Step1探针实测 |
|---|---:|
| 保存rollout | 32 |
| 实际length stop | 0/32 |
| 平均/最大response长度 | 519 / 877 token |
| 生成think标签 | 8/32 |
| PG loss | 0.178859 |
| 裁剪前grad norm | 14.820581 |
| Student / Teacher entropy | 0.511617 / 0.569601 |
| Top-16 overlap ratio | 0.674043 |
| Sign flip / normalized leakage | 0 / 0 |
| entropy/advantage/ratio非有限值计数 | 均为0 |

32条轨迹均以Llama原生BOS开头，输入保留历史要求think的指令，所有rollout
logprob为有限值。原文、原始token IDs、mask、sampling和原生停止原因均保留。
`diagnostics/scalars.jsonl`和`diagnostics/step_00001.npz`均已生成。

checkpoint含四rank各自的model、optim、extra_state及`data.pt`。实际读取
extra_state确认四rank均含scheduler和CPU/CUDA/NumPy/Python RNG，data.pt
含数据加载器snapshot。这不是仅保存HF权重。

### Step2

第二个探针PID637706于00:01:47启动，显式`resume_path`指向探针Step1，
使用`probe2`独立轨迹目录。四个rank的日志均确认从Step1加载model、optim
和extra_state，随后完成Step2并保存完整checkpoint。00:10:28开始最终探针
审计；探针及审计退出码均为0，acceptance包含checkpoint/diagnostic步骤
`[1, 2]`、`passed=true`，issues和warnings为空。

| 指标 | Step2恢复探针实测 |
|---|---:|
| 保存rollout / 实际length stop | 32 / 0 |
| 平均/最大response长度 | 507 / 601 token |
| PG loss / 裁剪前grad norm | 0.215275 / 19.745466 |
| Student / Teacher entropy | 0.710885 / 0.783391 |
| Top-16 overlap ratio | 0.696776 |
| Sign flip / normalized leakage | 0 / 0 |
| entropy/advantage/ratio非有限值计数 | 均为0 |

退出清理阶段出现`MathMultiProcessEnv.__del__`中的DataLoader worker killed
警告。它出现在Step2更新、checkpoint保存和final validation之后；进程正常
退出且产物审计通过，目前未发现其破坏该探针产物。保留原日志，不为消除
警告而热修改冻结运行版本；后续仍检查正式训练是否出现同类实际故障。

## 正式Token训练启动

正式训练PID647058于9月19日00:10:44北京启动。实际command和配置确认：

- `resume_mode=disable`，初始化原始`models/Llama-3.2-1B-Instruct`。
- teacher为原始`models/Llama-3.2-3B-Instruct`，不是历史Qwen教师。
- `total_training_steps=200`，保存里程碑50/100/200，无checkpoint删除策略。
- 训练协议`llama32_historical17_v1`，保留用户要求的历史think指令；评测
  协议仍为`llama32_nonthinking_v1`，这是用户明确要求复刻的训推差异。
- raw轨迹写入`token_opd/rollouts/formal`，不覆盖探针；诊断间隔5步，另保留
  Step1和50/100/200诊断。正式初始化不继承探针的两步更新。

00:13:58快照处于数据索引构建阶段，尚无正式rollout。00:18:38已完成正式
Step1，00:26:16已完成Step14，原始轨迹目录按步持续增加，没有观察到OOM、
NaN/Inf或进程失败。已有探针和训练过程结果不能当作benchmark对照结论。

正式Step1与探针Step1的32条prompt IDs、response IDs、mask、logprob、原题
信息和sampling参数逐条一致，支持原始初始化及首批输入一致。正式Step2与
重启后的探针Step2输出不同；恢复测试证明状态可加载并继续更新，不证明
vLLM采样在重启前后逐bit重放。这一边界与历史恢复测试一致。另实际对比了
正式Step2与恢复探针Step2，32条prompt IDs及原始题目index顺序全部一致：
题组index为476212、1167339、1464692、234773。输入数据进度没有漂移。

前两步的raw压缩文件SHA校验均通过，每步4题各8条；全部response长度、token
IDs长度和mask之和一致，logprob均为有限值。位置诊断已保存Step1/5/10。

| 正式步骤 | 平均长度 | length stop | 裁剪前grad norm |
|---|---:|---:|---:|
| 1 | 519 | 0/32 | 14.821 |
| 2 | 1339 | 0/32 | 15.035 |
| 3 | 4813 | 8/32 | 6.386 |
| 9 | 4593 | 8/32 | 8.813 |
| 12 | 1512.5 | 0/32 | 4.438 |
| 14 | 601.25 | 0/32 | 8.667 |
| 15 | 4594.5 | 8/32 | 7.681 |

截至Step15，实际length stop累计24/480=5%，裁剪前grad norm范围
4.438-27.173。此处以名义归档轨迹计数，组内输出重复限制见下节。

Step10 student/teacher entropy为0.720517/0.781368，Top-16 overlap为
0.694022；所有已保存诊断的非有限值计数为0，Token的sign flip/leakage为0。
这些数值不保证数学答案正确，也不能仅凭有限梯度排除输出质量退化。

## 已观察到的数据质量限制

- Step3一题的8条轨迹重复展开数学分析直到16384 token；抽查原文确认是生成
  重复，不是OOM或程序报错。Step9也出现一组8条length stop，后续数步未截断。
- Step1-8逐题检查，每题名义8条输出均只有1条不同的token序列，sampling
  seed均为21。不能将每步32条归档称为32个独立rollout。该现象已在
  `2026-09-13-qwen06-nonthinking-startup.md`记录；本轮沿用相同采样规则，
  不在Token训练中途改变seed、采样多样性或梯度权重。
- 这限制了有效采样多样性和统计解释，但不等于文件重复写入或mask损坏。
  若未来修正为不同采样随机流，应单独版本化并重跑两组，不能与本轮混合。

## 过程图快照

00:25使用冻结版本的`analyze_single_opd_diagnostics.py`生成快照，退出0。
远端目录为`token_opd/supervision_snapshots/20260919_0025`，未覆盖正式
`figures`目录。包含位置热图、entropy segments和3张标量图，已查看热图。
灰色区域表示没有有效token观测，不表示零熵；目前只有Step1/5/10位置诊断，
不能据此重建未采集诊断的Step3/9位置熵。所有步骤的raw轨迹仍完整保留。

本地副本：`/home/tyf/paper/outputs/llama-historical17-supervision/20260919_0025/`。

## Step25质量警告

00:38:39已完成Step25，归档25步共800条，实际length stop累计88/800=11%。
Step24、25分别有16/32条截断，不能仅用首批0截断描述整段训练。逐步日志未
发现OOM、NaN/Inf或任务退出；裁剪前grad norm范围2.568-27.173。
日志记录的CUDA峰值allocated约71.733 GiB、reserved约79.023 GiB，显存较紧，
继续观察；reserved不等同于实际活跃张量大小，也不等于已经OOM。

只读统计Step20位置NPZ，student entropy在前512位置为1.013996，在8K-16K
为0.017769；相同长尾上的teacher entropy为0.016190。Step15对应长尾学生/
教师熵为0.015473/0.002544。低批平均熵在这些批次中包含很低熵的重复长尾，
不能直接解释为所有位置共同退化，也不能据此认定教师在长尾上的指导正确。
这仍是动态、不同题目的训练批次观测，不是固定prompt纵向因果比较。

继续保持既定实验条件：生成质量退化需要忠实记录，不是偷偷换prompt/loss
的理由；真正的工程故障另行保留现场、定位和版本化修复。

## Step44原文抽查

01:04:09完成Step44，累计length stop为208/1408。抽查长输出发现：

- Step40，原始data index318565：反复重复同一段关于模p矩阵的分析，length stop。
- Step43，index86089：展开无止境的分数列表到16384 token，length stop。
- Step44，index712412：15090 token后以EOS结束，但后段出现无关文字和反复
  改写答案。因此无截断、数值有限都不足以证明输出正常。

这些都是**Token基线**观测，不是Block3独有的问题。该新师生组合的方法
对比仍待两组训练与完整评测，不能据此给Block3贴正面或负面结论。

按每题组计一次，粗分题面是否含U+4E00-U+9FFF汉字（不是语言分类器）：

| 训练区间 | 题面分组 | 题组数 | length stop题组 | 平均response长度 |
|---|---|---:|---:|---:|
| Step1-20 | 不含汉字 | 57 | 5 (8.77%) | 2123.72 |
| Step1-20 | 含汉字 | 23 | 1 (4.35%) | 1635.35 |
| Step21-44 | 不含汉字 | 71 | 11 (15.49%) | 3929.93 |
| Step21-44 | 含汉字 | 25 | 9 (36.00%) | 7209.32 |

这里只是不同批次的小样本描述，不是同题翻译对照，不证明语言造成退化。
两类题面在后段都更易出现长输出，不能只根据一个中文坏例下结论。

## Step50保存与阶段核验

01:16:40观察到Step50完成且checkpoint生成；01:19:04已完成Step52，说明
正式训练在保存之后继续推进。此次监督没有修改训练代码、loss、seed或
超参数；冻结运行版本的`.expected.sha256`复查退出0。

### 完整状态

实际CPU读取Step50的四rank优化器和extra_state（不占用训练GPU）：

- 每rank均有model、optim、extra_state；model文件非空且ZIP格式检查通过，
  本次未逐tensor重载模型权重。
- 每rank优化器17组state slots，已记录的optimizer step全部为50。
- 四rank的scheduler `last_epoch=50`，optimizer/scheduler学习率均为2e-6。
- 四rank RNG均含CPU、CUDA、NumPy和Python状态。
- `data.pt`的snapshot step为50，sampler已消费200条prompt，iterator未结束。
- `latest_checkpointed_iteration.txt=50`；tokenizer、chat template、模型config
  和generation config都存在。

这次检查没有中断正式训练去重新加载Step50模型；真实保存后恢复测试是在
前述两步探针中完成的，二者证据应分开表述。

### 前50步归档与数值

50个gzip文件逐个读取、SHA校验；样本编号、来源、长度/mask和logprob均通过。
总计1600条、200个题组，平均3834.555 token，length stop272/1600=17%。
每题8条仍只有1条不同token序列，共200条组内唯一输出，不能视作1600个
独立采样。没有缺步或损坏归档。

位置诊断步骤为1、5、10、15、20、25、30、35、40、45、50。全部已保存
entropy/advantage/ratio非有限值计数为0，Token sign flip/leakage为0。

| Step50指标 | 实测值 |
|---|---:|
| PG loss | 0.125464 |
| 裁剪前grad norm | 4.523863 |
| Student / Teacher entropy | 0.320389 / 0.291461 |
| Top-16 overlap ratio | 0.658197 |
| Student / Teacher overlap mass | 0.989326 / 0.991624 |
| 该步length stop | 8/32 |

前50步日志中的裁剪前grad norm范围2.044-27.173，未发现OOM、NaN/Inf或
进程错误。这不是生成质量或benchmark增益的验收：前述重复、无关文本和
截断均属真实质量警告，仍须完成Token/Block3同条件评测。

### 图表与后续状态

Step50图表脚本退出0，生成HTML和5张PNG，已实际查看位置热图：

- 远端：`token_opd/supervision_snapshots/20260919_step50/`。
- 本地：`/home/tyf/paper/outputs/llama-historical17-supervision/20260919_step50/`。
- 正式结束后的`figures`目录仍交由既定队列生成，两个监督快照不覆盖它。

01:19:04的最新状态为Token Step52/200，52个轨迹目录，累计280/1664条
length stop。Block3尚未开始。Token完整训练及三个里程碑评测在前，随后
执行Block3探针、训练和评测。后续状态应查实时日志，不能将本快照当作完成。

## 监督边界

第一批有限梯度和零截断只能说明该批没有观察到相关异常，不是长期稳定性
或方法有效性的证明。后续继续检查状态加载证据、逐步轨迹覆盖、过程指标、
checkpoints以及完整分benchmark评测。失败现场、已有评测与权重不删除。
