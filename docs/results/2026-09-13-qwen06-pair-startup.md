# Qwen3-0.6B 配对验证启动记录

## 最新变更：取消时间上限

用户随后明确“不用顾忌时间，放心做实验”。北京时间2026-09-13 01:05:29，
无时间上限控制器已启动，PID **3547500**、PPID=1，runtime
`ec0a7a950540d7f4753c08b18bc26c544f6a7cde`，日志为
`$RUN/logs/queue_no_deadline.log`。状态仍为`waiting_for_predecessor`，
`phase_started=false`、`budget_seconds=null`。正式GPU实验尚未开始。

替换前冻结旧等待控制器，确认尚无phase manifest或子作业，保留
`control_history/remove_time_cap_20260913/`中的原PID、状态及日志后，
只终止该等待进程。SIGTERM退出记录为143，是人工替换控制器，不是训练失败；
首次15秒退出轮询超时，随后确认进程消失、锁释放，才启动新控制器，未强杀。
旧窗口训练/评测没有中断。新旧runtime的1312个文件仅队列脚本不同，
训练、loss、eval、模型和数据未改。全套测试510通过、2跳过。

范围仍为0.6B Token OPD → 原版Block3 mean，各200步、三轮四任务完整n8评测。
取消时间硬停不等于允许自动追加师生、seed、训练步数或改变评分协议。

当前评测文件核查：MATH500500、AIME24 30、AIME25 30、AMC23 83；
各文件内部没有题面逐字相同的重复项。这不是训练/评测去污染证明。
AMC23的83条指历史冻结文件版本，论文需披露其来源与哈希，不仅凭名称
将成绩与其他版本直接等同。

## 原始启动记录

截至北京时间 **2026-09-13 00:52:45**，后台队列已启动，但状态是
`waiting_for_predecessor`。新0.6B实验尚未执行 GPU 门禁或正式训练，
无新得分；不能把部署、单测或配置预检写成实验成功。

## 批准范围与队列

- 方案：`docs/plans/2026-09-13-qwen06-paired-validation.md`。
- 仅 ml2，顺序 Token OPD 完整训练/三轮完整评测，然后原版 Block3 mean。
- Queue PID `3544990`，已确认 PPID=1，脱离 SSH 生命周期。
- Asset root：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
- Run：`$ROOT/runs/20260913v1_qwen06_pair_seed21_ml2`。
- 日志：`$RUN/logs/queue.log`；状态：`$RUN/queue_state.json`。
- Runtime：`$ROOT/deployments/937ac50f9fe3abb9f9bed4eee870c100b297b22a`。
- GitHub 分支：`feat/ml2-block3-replication`；runtime commit已推送。
- 前驱：`$ROOT/runs/20260911v2r1_sliding_window_seed21_ml2`，恢复队列
  PID3533800仍正常，正在 Sliding3 Step50 完整评测。
- 新预算尚未开始，`queue_manifest.json`尚不存在。前驱全部成功、释放锁后
  才写不可变起止时间并开始48个四卡机时预算，包含门禁/训练/合并/完整eval。

## 已完成的预检

1. 官方 `Qwen/Qwen3-0.6B-Base` revision
   `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，9个模型文件已下载并核对
   官方文件大小、Git blob/LFS哈希。权重 SHA
   `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`。
2. ml2直接联网失败，日志保留在`preparations/20260913_qwen06/download.log`。
   改为本地下载官方固定快照、校验后传输；没有换模型版本或修改权重。
3. 0.6B与历史1.7B学生的共有词表和模板一致。教师的4个历史额外标签已写进
   `models/Qwen3-0.6B-Base/asset_manifest.json`的`tokenizer_compatibility`。
   严格“完全相同”预检最初拒绝了该差异，调查确认是历史教师注册标签后，
   才加入固定ID的窄例外与回归测试。没有改teacher/tokenizer文件或loss mask。
4. 现有教师和数据18个文件与历史实验SHA一致。完整资产预检覆盖30个受保护文件，
   包括新学生、旧教师、数据、历史grader；均通过。
5. 在独立`preparations/20260913_qwen06/prepared_controls/`中真实执行
   `PREPARE_ONLY=true`，两组run-card及资产清单比较通过。只允许五项不同：
   `variant`、`opd_block_size`、`opd_block_advantage_mode`、`experiment_name`、
   `diagnostic_output_dir`。它们是配置预检产物，不是正式run。
6. 新版本1312文件校验通过，旧窗口runtime的1274文件仍全部通过原校验；
   没有覆盖旧release，也没有改`deployments/current`链接。
7. 全套测试508 passed、2 skipped；Bash语法及Python编译通过。独立复审
   发现并修复诊断目录继承、残留子进程和deadline覆盖问题；二次复审无剩余发现。

## 尚待队列实际验证

- 两方法的完整形状 Step1保存/退出/恢复Step2门禁。
- 各自从同一Base独立初始化的200步训练与41次诊断快照。
- 各 Step50/100/200 四任务643题×8回答，主历史grader不变。
- 最终配对结果、prompt hash一致性、bootstrap区间与热图。

失败时保留日志/checkpoint，关闭队列，不自动调参、换seed、缩eval或重置预算。
最终结论仅能讨论这一个training seed、同一教师下的学生容量泛化。
