# 2026-09-18 历史 1.7B 重评与 Llama 队列重排

## 用户决定

先停止正在运行的 Llama Token 训练，重评历史 Qwen3-1.7B Token / Block3 的
Step50、100、200 共六份权重，保存全部评测 rollout。随后重新开始 Llama
对照，并明确复刻旧 1.7B 的训推提示差异：训练要求 think，评测不要求。
Llama 仍用原生聊天模板、固定日期和原生 EOS，不使用 Qwen 特殊 token。

方案见 `docs/plans/2026-09-18-historical17-reeval-llama.md`。

## 已完成的停止操作

- 仅访问 ml2，没有访问 train。
- 核对 controller297837 的实际命令后发送 SIGTERM；控制器只清理自己
  创建的当前训练进程组，没有执行全局 Ray/GPU kill。
- 旧队列在 2026-09-18 12:01:41 北京时间记录 `failed/error=143`，这是用户
  主动停止，不是自发训练故障。其最后完成的更新为 Step146。
- Step50、100 完整 checkpoint 及146步 rollout 存档保留。未生成 Step200。
  Block3 未开始；Llama 初始 Instruct 学生完整评测已完成。
- 停止后实测四张 GPU 均为0%利用率、4MiB显存。没有删除原始产物。

## 新实验边界

- 新目录：`runs/20260918v2_historical17_reeval_llama_seed21_ml2`。
- 六权重来自 `20260712v1_token_opd_replication_seed21_ml2/token_opd` 和
  `20260711v2_block3_replication_seed21_ml2/block3_mean`，不重新训练 Qwen。
- 保留历史温度1、top-p0.9、max_tokens16384、n8、seed21-28、四GPU分片
  和原始评测 prompt。评分使用历史最终对照采用的 external grader
  SHA256 `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
  原生成配置的 builtin verl 分数不代替最终历史 external 重评分视图。
- Llama Token / Block3 都从 ModelScope 原始1B-Instruct重新初始化，不从旧
  Step100、已停止的内存状态或 probe 权重续训。保持 seed21、同数据、200步
  和完整四任务 n8 评测；新增版本化历史训练提示协议。
- 新评测逐 benchmark 独立计分，保留 grader 接收的原文本以及特殊 token
  不过滤的原文、原生 token IDs 和 engine finish/stop reason。
- 重生成样本不能冒充旧训练轨迹；相同参数不能保证跨运行时逐字复现。

## 启动状态

正在实现和验证新不可变运行时；本节不代表新评测或后续 Llama 已启动。
实际启动 PID、release 和首批 GPU 证据在启动后补充。
