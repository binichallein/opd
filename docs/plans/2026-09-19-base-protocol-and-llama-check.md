# Base 新协议与 Llama 格式排查

## 用户已确认的修改

1. 下一轮 Base 师生不用 ChatML，不要求 think 标签。训练与评测共享 completion 提示。
2. 全局 seed 保持21；请求 seed 从 step、题目及 rollout 序号稳定派生，两组同规则。
   不能使用进程随机化的 Python hash，也不能加入 Token/Block3 组名使对照分叉。
3. 先用原始学生和教师做固定训练题 GPU 推理验收，记录原始输出、重复、截断、
   答案格式和教师答题质量，再建新版本正式实验。不能用低截断率替代答案质量。

候选提示为题目加“Please solve the problem step by step and put the final answer in
\\boxed{}.”，最后接 `Solution:`。这不是之前已测的裸题目加 Solution，尚待精确验收。
保留数据、grader、分 benchmark 的完整评测、算法及其他超参。新实验仍要求保存
50/100/150/200完整状态和所有 rollout。旧 Qwen4 v1 停在Step4，保持停止，不覆盖证据。

## 当前先做的 Llama 排查

- 原始 ModelScope Llama-3.2-1B-Instruct 和 3B-Instruct，不是 Base。
- 原生 BOS128000、header128006/128007、EOS/EOM/EOT128001/128008/128009；
  不使用 Qwen 控制 token，也不把 Llama 原生 chat 错叫 ChatML。
- 取旧 Token 训练Step1-4首次出现的16道题，只复用问题和实际输入，不把Step2以后的
  已更新模型回答当原始模型输出。固定顺序，未经按表现筛选。
- 每题2次，请求seed21/22，四种提示，两个原始模型，共256条，max tokens16384。
  温度1、top-p0.9、top-k-1、原生停止ID，vLLM0.11.0，BF16，TP1。
- 四种提示：历史原生chat+think要求；只删除think要求；同一去think的用户内容改为
  BOS+completion+Solution；实际历史评测用的原生chat提示（无think要求）。
- 第二到第三组不仅删除边界token，还移除system/header文字并加Solution，属于模板
  整体干预，不能声称是单个边界token的因果效应。
- 历史组核对实际输入token IDs；保持模板日期18 Sep 2026，不随当天漂移。
- 记录引擎真实finish reason、全部token/logprob、末2048 token周期性和4-gram重复、
  boxed标记/think标签。boxed出现不是历史format-error定义，也不等于答案正确。
- 原始模型结果结合历史首批及训练后熵/输出退化时间线分析；不凭这次短诊断判定
  Block3 loss的具体因果机制，也不将16题当作完整benchmark结果。
- ml2独立诊断目录，后台运行且保存命令/PID/日志/模型和代码哈希；不修改冻结部署。
