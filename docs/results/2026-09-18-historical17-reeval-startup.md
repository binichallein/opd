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

- 2026-09-18 12:32:57 北京时间，在 ml2 通过 `nohup` 启动唯一新控制器，
  PID `434226`。不可变运行时：`94be7ea1d659309256c8356681925bb9710895c4`。
- 命令：上述部署目录内的 verl 环境 Python 执行
  `-u scripts/run_historical17_reeval_llama.py`。控制器日志为新目录内
  `logs/controller.log`；实时进度以 `queue_state.json` 为准。
- 启动时再次确认旧 controller/train PID 均不存在，四张 GPU 均为0%/4MiB。
  12:33 的状态为 `preflight`，部署文件哈希检查已完成，正在核验保护输入。
  此时尚不能宣称评测生成或后续 Llama 已开始。
- 评测顺序：Token50、Block3-50、Token100、Block3-100、Token200、Block3-200。
  六项全量验收通过后，才进入 Llama 实际 GPU 提示检查、初始学生评测、
  Token 训练与评测、Block3 训练与评测。
- 12:39:43 首项 `qwen17/evaluations/token_opd_step50` 已启动，eval PID
  `435131`，状态 `running`。12:41 四个 worker 均已加载正确权重、进入
  编译/预热，四张 GPU 各占3919MiB；没有提前调度 Llama。
- 实际 `outputs/eval_config.json` 已确认完整四任务、n8、seed21-28、温度1、
  top-p0.9、16384上限、external grader、`enable_thinking=false` 和
  `retain_rollouts=true`。这仅证明任务按配置启动，不是全量评测完成或得分。

## 验证记录

- 本地最终完整测试复跑：657 passed、2 skipped，用时19.51秒。独立代码
  审查发现的 ModelScope revision 选择问题已修复并回归。
- ml2 不可变部署没有 `.git`。第一次全套测试中641项通过，15项失败均来自
  依赖 Git 工作树的旧队列测试文件；另一个依赖历史 `git show` 的测试被排除。
  这些 Git 相关测试已在本地有仓库的环境通过，不属于生成/训练运行时故障。
- 排除该 Git 依赖文件和上述一项历史 Git 测试后，ml2 验证结果为
  **611 passed、2 skipped、1 deselected**，用时20.43秒，无 pytest 磁盘缓存。
- ml2 原始 Llama 1B/3B tokenizer 的 CPU 提示检查均生成了
  `analyses/20260918_historical17_cpu_preflight/94be7ea/{student,teacher}/input_contract.json`。
  训练标记为 `enable_thinking: null`，评测为 false，明确记录输入不同。
  CPU 检查不能代替 GPU 检查；GPU 检查仍按队列在六权重重评之后执行。
- TOS 上的逐条轨迹持久化所需目录 fsync 已实测可用。存档按实际返回的
  generation batch 增量写入；尚未生成完成的 batch 不能承诺断电可恢复。
- 启动后六个历史权重的实际输入检查全部通过，各覆盖643题，输入哈希均为
  `af456ee536c0c0f75eb23311ec94e33020c384074993a13ae302e6a11f57dac9`。
  `protected_inputs.json` 已登记615个文件。
