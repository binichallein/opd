# Qwen4 Completion Token OPD 对照训练

用户要求先准备Token训练代码，当前Block3四个权重的完整评测结束后自动训练。
本次授权覆盖新的Token训练队列；不修改、暂停或重启已有训练/评测队列。

## 对照目标

对照已经完成的 `20260920v3_qwen4_completion_blockfirst_seed21_ml2/block3_mean`。
沿用其所有公共训练设置和冻结实现，只将方法切换为原始sampled-token OPD：
`variant=token_opd`、`opd_block_size=1`、`opd_block_advantage_mode=sum`。
单token的sum与mean等价，不做邻居advantage共享；使用该实现原有的逐token PPO ratio和归一化。
Block3保持历史联合block ratio及block归一化，此对照不是“只改变advantage共享”的单因素消融。

## 固定配置

- 仅ml2四张A100；不访问train，不使用私人权重，不重新下载或换模型。
- 学生：官方ModelScope `Qwen/Qwen3-4B-Base`，revision
  `bbd6fc8d23e8788d987b7b970cbb7bd31c826e38`。
- 教师：同一公开 `Qwen3-4B-Base-GRPO`；模型字节与Block3的artifact manifest逐项对齐。
- 从原始学生初始化，`resume_mode=disable`，不续训Block3、旧ChatML或探针权重。
- 原DAPO-Math-17K扩展池1,791,700行，train.parquet SHA256：
  `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`。
- 全局/data/environment seed21，独立请求seed规则 `sha256_step_question_sample_v1`。
  同step、同题、同sample的请求seed与Block3对齐；模型更新后输出不同是正常现象。
- 输入 `qwen3_completion_boxed_v1`，无ChatML，不要求think标签，显式thinking=False；
  EOS151643、历史EOS mask；后续评测也必须显式使用此completion协议。
- 每步4题，每题8条，共32条；200步，LR2e-6，PPO epochs1，mini-batch32。
- actor/logprob/ref micro-batch=1/4/1；vLLM memory0.6。
- prompt上限2048、response上限16384，temperature1.0、top_p0.9、top_k=-1。
- 保存Step50/100/150/200完整模型、四rank优化器/调度器/RNG、dataloader状态及配置；全部保留。
- 每步保存实际32条rollout、输入/输出token、logprob、请求seed与真实终止原因。
- 每步保存师生熵与熵差、Top16重合率与概率质量、overlap advantage、sign flip/leakage、
  PG loss、裁剪前grad norm、长度/截断及位置NPZ；训练结束生成热图。

## 启动条件与保护

- 新控制器 `scripts/run_qwen4_completion_token.py`，独立immutable analysis deployment。
- 新实验目录：`runs/20260920v1_qwen4_completion_token_seed21_ml2/token_opd`。
- 训练实现继续冻结在 `deployments/0f9161f02f08287fb07f0375ad0a6bda81133ff0`。
- 等待 `20260920v1_qwen4_completion_block3_eval_seed21_ml2` 完成全部200/150/100/50评测，
  每个权重643题、5144条、32个原始archive通过验收，历史grader和prompt合同通过。
- 前置队列失败、退出非0、轨迹缺失、模型/数据/实现hash不一致时禁止训练，不自动重试。
- GPU空闲后才准备正式command；复用此前Token真实两步更新/恢复和四rank优化器验收。
- run card和实际command与已完成Block3严格对照；只允许方法、实验名及输出/cache路径变化。
- 临时编译缓存使用独立可执行tmpfs `/dev/shm/opd-q4-t1`，持久化产物仍在实验目录。
- 服务端nohup，不依赖本地终端；不删除checkpoint、不全局停止Ray，不覆盖原始结果。
- 完成训练后检查四个checkpoint、200步诊断与全部6400条轨迹，并逐step与Block3核对题目、
  输入token及请求seed。无需两组输出文本相同，也不声称vLLM按位复现。
- 此控制器训练后只验收并画图，不自动插入初始模型评测或Token benchmark评测。
  后续完整评测仍按MATH500、AIME24、AIME25、AMC23分别计分，不用macro；Step200为主终点。
