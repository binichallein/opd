# Qwen4 初始输出重复循环排查

## 用户停止

2026-09-19 21:32 北京时间，按用户指令停止本轮训练和整个后续队列。
仅向已核对的控制器 PID1047903 发送 SIGTERM，由其清理本任务进程组。
21:33 确认控制器退出，4张GPU均为4MiB、0%利用率，无后续任务启动。
队列原始状态为 failed/error143，含义为本次主动停止，不是自发训练故障。

- 正式训练已完成 Step4，4批原始rollout保留。
- 正式checkpoint尚未到约定Step50，因此没有正式保存权重。
- 两个恢复探针的完整checkpoint、所有历史日志和配置保留，不清理、不覆盖。
- 正式runtime与配置不变；不会自动恢复训练或启动排队评测。

## 诊断设计

只用原始模型推理，不用探针或正式训练权重。独立目录保留脚本、输入token、
采样seed、完整response、结束原因、sampled log-prob、版本和文件哈希。

1. 用存档首批4道题的真实prompt IDs，独立vLLM重放，隔离训练框架、FSDP权重同步。
2. 固定每请求seed21与同题独立seed21..28对照，区分同题重复与输出内部循环。
3. 保持模型和采样不变，去掉think指令，再比较裸题目加Solution提示。
4. 同样输入和解码在原始1.7B上运行，检查模型差异。
5. 初筛使用1024上限，识别早期循环；不能把此处length stop当成正式16384截断率。
   根据结果用16384复核关键单元；若独立vLLM与训练差异大，进一步用HF GPU复核。

不预设Base/seed/vLLM哪一项是原因。重复检测沿用已有脚本的末尾周期与4-gram指标，
算法检测加人工阅读，不把长度上限、解题错误和重复循环当成同一指标。

## 已确认的证据链

### 1. 不是必须经过训练才发生

正式首批在第一次优化前采样，两个循环输出长16384，名义上各重复8次。
独立vLLM直接加载同一原始4B，未经过FSDP、权重同步或optimizer，也复现杂乱开头与循环。
独立回放的具体序列/截断数与正式训练不完全相同；不能把两套结果混记，
也不能用“成功复现现象”声称后端之间逐位一致或所有实现问题均已排除。

### 2. 聊天格式与原始Base的适配是已验证的诱因

[Qwen官方说明](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#naming)
明确区分Base和遵循预定义聊天模板的后训练模型。tokenizer存在chat_template，
不代表原始Base权重已经学会该协议；“与旧1.7B模板一致”不是“新模型生成质量合格”。

实际历史输入为以下结构，think要求在user正文，不是在assistant开头强制插入：

```text
<|im_start|>user
Math problem: [question]

Please carefully reason through the math problem step by step and derive the correct answer. You must conduct reasoning inside <think> and </think> and give the final answer within \boxed{}.
<|im_end|>
<|im_start|>assistant
```

严格的marker干预仅从原始IDs删除两个151644和一个151645，所有其他token原样保留，
包括user/assistant文字、题目、think指令。没有换数据、权重、采样参数或EOS。

CPU Transformers/BF16/SDPA的4B首token分布对比如下。下表不是生成长度统计，
而是同一个位置的top-p=0.9候选词表大小：

| 数据index | 历史ChatML | 仅删3个marker | 裸题目+Solution |
|---|---:|---:|---:|
| 1434553 | 7751 | 65 | 44 |
| 1742959 | 8863 | 81 | 14 |
| 69824 | 7656 | 245 | 71 |
| 1225498 | 6043 | 101 | 46 |

1.7B同样出现此现象：前三题候选数6552/7331/7021，删marker后为17/17/177。
这不是“4B能力一定弱于1.7B”，而是原始模型对当前提示的适配问题。

### 3. 权重层的补充解释，不冒充初始化历史证明

直接读取官方原始safetensors，两模型均tie input/output embeddings。
以4B的151644输入向量为参照，296行逐值相同，2227行满足每个分量最大绝对差
不超过1e-4。151645与参照的L2差为1.53e-5，实际采到的125576为1.91e-6；
151667/151668也与参照相同。1.7B在相同阈值下分别为94/297行。

这些是明确阈值下的向量相似性，不能据此声称已证明其初始化方式、每个token的
训练历史，或所有乱码都由这个子集解释。4B前两题中，这2227个token的首步原始
总概率为4.57%/3.84%，top-p后为5.07%/4.26%；仅删marker后均被排出nucleus。
它解释了为什么低概率杂乱token仍可合法进入候选集，不能简单归咎于top-p失效。

### 4. 进入循环后，模型自身强烈偏好继续重复

正式首批两个循环末尾1024token的平均被选token概率为99.94596%和99.98674%。
16条截断输出内既没有EOS151643，也没有im_end151645，不是已输出结束符但被忽略。

额外独立HF GPU验证：原始4B接上两条实际长前缀，不通过vLLM，用greedy再生成32token，
仍分别复现短语循环和数值循环。两个位置对原来被选token的预测概率为
0.99997568/0.99992573，EOS概率为3.77e-8/4.54e-5。
这证明模型分布本身支持继续循环；不把它说成概率严格为1的吸收状态。

### 5. 固定请求seed是重复放大因素，不是唯一根因

历史每请求seed21使同题输出高度重复。正式首批只有4种输出，不能当作32次独立失败。
独立推理同seed也存在少量因数值/批处理差异产生的分叉，不能承诺恒定逐位重复。
全局实验seed固定用于复现，与每个请求反复初始化成同一个seed，是不同问题。

改成每题seeds21..28后，4B仍有循环；1.7B也暴露循环。此前1.7B在seed21上
能结束不等于原始输出质量健康：部分未截断输出本身已有杂乱开头或错误解答。

## 完整长度对照

同一批4道训练题，每题8条，temperature1/top-p0.9/top-k-1/max16384/EOS151643。
以下均为独立vLLM的length stop，不是benchmark成绩或整个DAPO训练集的估计。
fixed组高度相关，independent组是每题seeds21..28，不能把两者当作等量独立样本。

| 输入与采样 | 4B截断/32 | 1.7B截断/32 |
|---|---:|---:|
| 历史ChatML，fixed21 | 8 | 0 |
| 历史ChatML，independent21..28 | 8 | 8 |
| 仅删think指令，independent | 13 | 11 |
| 仅删3个聊天marker，fixed21 | 8 | 0 |
| 仅删3个聊天marker，independent | 4 | 2 |
| 裸题目+Solution，independent | 2 | 1 |

删除marker后仍有数学表达式重复。4B fixed组的向量题仍到上限，末尾4-gram重复率
约97.9%，但严格周期检测没有命中；因此periodic_tails=0不能声称“没有循环”。
须保留截断、周期检测、重复n-gram和人工阅读之间的区别。

贪心额外对照每组仅4条，不并入上表：历史ChatML的1.7B为4/4到上限；
删除marker后1.7B为1/4、4B为0/4。4B历史ChatML为0/4。
贪心不一定是可用修复，不能把降低随机性当作解决全部重复的保证。

## 下一步边界

主要结论是原始Base与历史ChatML不适配，随机采样可进入坏上下文并形成模型自身
支持的重复，固定请求seed又将同一坏轨迹重复多次。不是Block3更新导致首批异常。
但提示改进没有证明可以消除所有数学循环，也没有验证Block3在新协议下的效果。

当前仅诊断，不恢复训练。新协议应先验收原始学生和教师在候选输入下的质量，
再让Token/Block3从同一原始权重、相同提示与独立派生的请求seed做新对照。
不能只给一个训练组修提示或seed，也不覆盖历史实验。另一条可比较路线是共享SFT
cold start，但那是新的初始化条件，不属于本次已验证的修复。

历史1.7B涨点仍是其原条件下的观测；本次不能直接宣布它无效，也不能把其中
格式适配的收益全部解释成推理信用分配改进。需要后续控制实验区分机制。

## 产物

远端根目录：`opd/diagnostics/20260919_qwen4_initial_loop`。

- `q4_screen`、`q17_screen`：1024上限初筛，各132条。
- `q4_full`、`q17_full`：16384完整上限，各132条，均已完整结束。
- `q4_markers`、`q17_markers`：marker-only完整上限，各68条。
- `q4_alias.jsonl`、`q17_alias.jsonl`：CPU权重与12个首token分布记录。
- `q4_hf_gpu.jsonl`：GPU首token对照及两个长前缀的HF续写。
- `code`、`code_v2`、`code_v3`：各阶段实际脚本及SHA，不覆盖先前版本。
- 全部raw包含实际prompt/response IDs、文本、seed、结束原因、被选token log-prob。

训练代码和冻结runtime未修改；新诊断脚本的定向CPU测试14项通过。

22:08最后一组生成结束，22:09全部GPU空闲。22:10产物复核通过：
664条vLLM诊断输出完整，raw SHA、请求ID唯一性、输入/seed、log-prob长度与有限性、
summary统计均一致；两模型各个对照的prompt IDs与请求seed逐项一致。
CPU两组各12条首token记录，GPU12条首token记录和2条独立长前缀续写完整。
完整数字与raw文件SHA见[机器可读汇总](2026-09-19-qwen4-initial-loop-summary.json)。
固定seed输出不能当作独立样本，664是实际保存条数，不是独立观测数。

部分独立vLLM任务退出时有NCCL进程组未显式destroy的告警。所有预期产物均完整，
GPU已释放；不隐藏告警，也不把它误称为训练中OOM。正式训练仍停在Step4，
未启动后续评测/Token任务，未静默修改任何科学参数或重新开始训练。
