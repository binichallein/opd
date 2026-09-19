# Qwen4 Block3-First 启动记录

## 范围与身份

用户2026-09-19要求将新4B学生实验改为Block3先执行，并为两组增加Step150。
科学合同见[计划](../plans/2026-09-19-qwen4-blockfirst.md)。

- Student：ModelScope `Qwen/Qwen3-4B-Base`，固定revision
  `bbd6fc8d23e8788d987b7b970cbb7bd31c826e38`，三分片原始BF16。
- Teacher：原有公共 `Qwen3-4B-Base-GRPO`，用历史1.7B Token资产清单核验，
  不替换教师，不使用私有模型。
- 顺序：Block3训练200步及200/150/100/50完整评测 -> 原始学生完整评测 ->
  Token训练200步及200/150/100/50完整评测。两组分别先通过两步恢复门禁。
- 每组独立从同一原始学生开始。初始评测虽后执行，不能指向Block3训练后权重。
- 两组seed21、同一数据文件/顺序、历史1.7B提示与EOS-mask设置、LR2e-6、
  batch4x8 nominal rollout、response16384、micro1/4/1、vLLM0.6；无新SFT。
- 完整保留50/100/150/200模型、优化器、scheduler、RNG、dataloader及所有raw轨迹。
- 四benchmark各自完整n8历史grader计分，共9个模型视图46,296条评测输出。

## 已验证的实现

- 完整CPU回归：721 passed、2 skipped；新资产/队列/协议测试58 passed。
- 新协议`qwen3_historical17_v1`保持历史Math模板及未设置thinking的chat渲染；
  eval仍显式false。保存生成长度与完整训练张量mask，避免将原EOS mask悄悄
  改为length mask。已有非thinking和Llama协议保持原行为。
- `core_algos.py` SHA与上一轮部署一致：
  `6b0d53d1299baa1c8cacec6ec1b52cf59f843c9a27fb40e4c903bbdbc4bffa71`。
  Block3依然是mean advantage + joint ratio + 原block reduction。
- 更新后的patch可反向核验，1248项上游runtime清单通过。旧部署不变。
- 独立复核发现并修复：磁盘预算不足、轨迹验收未锁定top-k/EOS/宽度。
  新门禁要求初始1TB，并按实际探针checkpoint体积预算后续完整保存与合并开销。
  EOS依据固定tokenizer核验（此官方Base为151643），不使用示例假定值。
- ModelScope per-file Revision代表最后修改该文件的commit，不等于snapshot
  revision。已用真实API验证固定快照13个文件哈希，避免错误拒绝合法文件。

## 部署与启动

- GitHub分支：`feat/qwen4-blockfirst`。
- 下载代码release：`ec005036bdef8b11a3875e17aa6ec29d77556502`。
- 正式队列/训练/eval冻结release：`57ae7dfec10e206dd441a4c70571308c09f64a03`。
- ml2 ROOT：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
- Run：`runs/20260919v1_qwen4_blockfirst_seed21_ml2`。
- 下载PID1044848已结束，`assets.log`确认`passed=true, files_verified=31`。
- 服务器nohup启动 `bash scripts/start_qwen4_pair.sh`，bootstrap/queue PID1047903。
  日志`queue.log`；从部署路径调用，不改旧current链接、不连接train。
- 北京时间20:29:32快照：`queue_state.status=preflight`，runtime清单验收完成，
  正在核验前驱/数据/资产。四GPU当时空闲，正式训练尚未开始。
- 20:31:19配对预检`passed=true`：两组确认为同一原始4B-Base和历史教师，
  保存列表均为50/100/150/200、恢复关闭、协议相同；仅方法及对应输出目录不同。
  预生成Token配置不是提前训练Token，实际执行仍以Block3开始。
- 20:31:58实际启动Block3 probe1，PID1049174；20:32:36 Ray启动成功，
  20:32:57快照仍为模型/worker初始化，尚未取得Step1保存或恢复通过证据。

这是启动阶段证据，不是恢复通过、正式训练完成或方法有效性结论。
后续必须读取live `queue_state.json`，不得依据本文重复启动队列。
全部checkpoint和失败现场保留；任何OOM/配置偏差/验收失败均停止，不自动换参或重试。
