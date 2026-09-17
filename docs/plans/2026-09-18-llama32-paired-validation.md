# Llama 3.2 配对验证执行计划

**Goal:** 当前 Qwen06 Block3 完整评测成功结束后，仅在 ml2 上完成 Llama-3.2-1B-Instruct 零步评测、Token OPD 训练与评测、Block3 mean 训练与评测。

**Architecture:** 新工作树、新不可变部署、新运行目录和独立等待控制器；复用 Revisiting OPD、已有训练/评测审计与历史 grader。仅增加 Llama 原生 prompt/EOS 适配，不改变 loss、数据或评分算法。

**Tech Stack:** PyTorch/verl/vLLM、ModelScope HTTP API、pytest、现有四卡 A100 环境。

## 固定实验协议

- 学生 `LLM-Research/Llama-3.2-1B-Instruct`，ModelScope revision `d3e551343d4d81508a0d226656b826c217e463cd`。
- 教师 `LLM-Research/Llama-3.2-3B-Instruct`，revision `4e7231b81c151c73632184994ac9a0149fcb22fd`。原开发者 Meta；LLM-Research 是镜像上传者，不是 Meta 官方账号。
- 两者为原始 BF16 Instruct。逐文件核对大小/SHA256、完整词表/模板一致性；不使用私有模型或 Qwen 权重。两者已有蒸馏历史，不声称它们是完全独立预训练模型。
- `base` 指上述学生的零步 Instruct 初始化。两条正式训练均从它独立开始，不继承 probe 或 Token 权重。
- 原 DAPO 文件 `data/math_opd_dapo17k_hf_full_eval4/train.parquet`，SHA256 `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`；实际 1,791,700 行，并非 17k 个唯一训练样本。保留原始顺序/数据采样实现。
- seed21、200步、每步4个 prompt、每题名义8个 rollout、LR2e-6、PPO epoch1、micro-batch1、vLLM0.6、prompt2048/response16384、temperature1/top_p0.9。
- Token: k1/sum；Block3: 固定不重叠 k3/mean。两臂仅聚合方式及各自输出目录不同。历史组内 rollout seed 重复行为保留，不能把8条称为8个独立样本。
- 相同题目指令及 `\\boxed{}` 要求，Llama 原生聊天模板；显式传 false，不注入 Qwen think 前缀。停止符按原生 generation_config `[128001,128008,128009]`；训练与评测一致。普通文字推理不等于 thinking 模式。
- 四个 benchmark 各自计分：MATH500(500)、AIME24(30)、AIME25(30)、AMC23(83)，每题8次，seeds21..28，temperature1/top_p0.9/max16384。历史 grader SHA256 `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
- 保存50/100/200完整训练状态，每个 checkpoint 完整评测。每次评测643题/5144回答；零步加两臂三个 checkpoint 共7次完整学生评测，不用抽样替代。
- 每步保存全部训练 rollout；保留原 entropy、gap、top16 overlap/mass/advantage、PG loss、裁剪前梯度、长度/截断、advantage sign-flip等诊断与位置热图。format error 继续定义为缺少 `\\boxed`，不混同解析失败。

## 执行任务

1. [ ] 测试先行，新增 Llama 模板/停止符适配；Qwen 回归测试不变。
2. [ ] ModelScope 固定 revision 下载及文件/词表核验，不占 GPU。
3. [ ] 新控制器等待 Sep17v2 queue_state=complete、所有三步完整评测验收及旧输入保护验证；失败/中断则停止，不抢卡、不自动重试。
4. [ ] GPU 空闲后核对实际训练与评测 prompt IDs；学生/教师各16题 GPU 推理测试，无新 think tag；任何失败保留证据并阻断正式训练。
5. [ ] 零步全评测；Token 两步断点恢复测试后正式200步及全评测；Block3 从同一初始权重重复相同步骤。训练前比较两臂 run card、数据和模型哈希。
6. [ ] 完整轨迹/checkpoint/指标/逐题评测审计、图表及各 benchmark 独立对照报告；训练过程质量下降是实验结果，不因低分删改。
7. [ ] 测试、复查、GitHub 同步、不可变部署、启动等待队列并核对 PID/日志/无 GPU 占用。

## 验收与边界

先通过 CPU 单测、补丁应用检查与 ModelScope资产检查；等待队列必须真实启动并留存命令、commit、PID、状态。GPU兼容性和训练成功只能在等待解除后确认，准备完成不等于这些门禁已经通过。无新 seed、无改数据/评分器、无训练时长上限、无 checkpoint 清理。只连接 ml2，不访问 train。
