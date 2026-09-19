# Qwen4 Block3 Mean 新协议正式训练

用户授权按新配置重启4B师生对Block3 Mean，并保存四个权重。

## 固定设置

- ml2四张A100；学生官方ModelScope `Qwen/Qwen3-4B-Base`，revision
  `bbd6fc8d23e8788d987b7b970cbb7bd31c826e38`。
- 教师沿用公共 `Qwen3-4B-Base-GRPO`，不使用私人模型或探针权重。
- 从原始学生重新训练，`resume_mode=disable`；不续训旧ChatML实验或两步验收权重。
- 原DAPO-Math-17K扩展池1791700行，SHA256
  `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`。
- 全局seed21；每步4题，每题8条轨迹；独立请求seed规则
  `sha256_step_question_sample_v1`，与刚完成的两组验收相同。
- `qwen3_completion_boxed_v1`：裸题目、step-by-step/boxed要求、Solution结尾；
  没有ChatML或think标记，显式关闭thinking；后续评测必须使用同一协议。
- Block3 Mean保持历史联合block PPO ratio与loss归一化，不是新改写的token ratio版本。
- LR2e-6，200步，PPO epochs1，mini-batch32；actor/logprob/ref micro-batch1/4/1。
- 温度1、top-p0.9、top-k-1；prompt2048、response16384；vLLM memory0.6。

## 保存与观测

- **Step50、100、150、200** 各保存完整四rank模型、优化器、调度器和RNG状态，
  以及tokenizer/config、dataloader游标；保留全部，不自动删除，最终200独立保存。
- 每步完整保存32条rollout，包括输入/输出token、logprob、真实请求seed和终止原因。
- 每步保存既有OPD诊断标量及位置NPZ：师生熵、熵差、Top16重合与概率质量、
  overlap advantage、sign flip/leakage、梯度、PG loss、长度与截断等。
- 训练结束自动检查四个checkpoint与200步诊断，再生成热图。
- 本次只启动Block3；不自动启动Token训练或benchmark评测。

## 版本与边界

- 训练仍使用已通过真实GPU更新/恢复的冻结版本
  `deployments/0f9161f02f08287fb07f0375ad0a6bda81133ff0`。
- 新控制器 `scripts/run_qwen4_completion_block3.py` 只负责启动、校验和收尾。
- 新目录：ml2 `runs/20260920v2_qwen4_completion_blockfirst_seed21_ml2`。
- 缓存短路径：`/limx_embap/tos/q4/c2`，避免Ray socket路径过长。
- 两组验收均完整恢复通过，复用该验收，不再额外运行一套重复探针。
- 验收中仍有低熵重复，不能保证正式训练不崩溃。禁止把新协议结果与旧协议Token
  直接当成只改变算法的对照；后续Token对照需复用本次全部公共设置。
