# Qwen3-0.6B 配对验证启动记录

## 06:45巡检：正式Token通过Step50保存检查

正式训练PID3594635已完成Step50并继续后续训练，Block3仍未启动。
没有更改任何对照条件，没有缩减评测或把probe结果计入正式实验。

- `global_step_50`包含四rank的model/optim/extra_state共12个非空文件，
  另有`data.pt`和模型/tokenizer配置。实际读取四rank extra_state及data.pt，
  scheduler步数均为50，RNG字段完整，dataloader消费位置为50。
- 正式日志Step1..50无缺步/重复步；11个诊断JSON记录及对应NPZ完整可读，
  step与prompt哈希对应，全部标量非有限、上溢、下溢检查通过。
  此时正式日志没有Traceback、CUDA OOM或DataLoader worker killed。
- 每步日志的裁剪前grad norm最小0.568、中位数8.386、P95约74.120、
  最大144.305。Step41与47分别为144.305、130.734，随后回落；
  这是尖峰，不足以据此认定持续梯度爆炸。统计使用日志的三位小数精度。
- 前50步逐批平均截断率33.5%，最初10步60%，最后10步40%；
  Step4、8、14全批截断。Step50本批为75%，grad norm8.17324，
  student/teacher entropy分别0.096077/0.030132。
  不同step使用不同prompt，不能把这些批次差异直接解释为学习增益。

独立CPU分析快照：`$ROOT/analyses/20260913_qwen06_token_step50_supervision/`，
`snapshot_manifest.json`记录来源、范围及复制后文件SHA。
复用冻结版本的绘图脚本生成5张PNG与HTML，没有改正式run或占用训练GPU。
本地注释版：`/home/tyf/paper/outputs/qwen06-token-step50-supervision/figures/diagnostics.html`。
远端保留原始生成HTML，注释版另存于同一分析目录的
`figures/diagnostics_annotated.html`，没有覆盖原始报告或诊断输入。
已检查热图和优化曲线非空。注释版纠正旧模板硬编码的Block3名称，并明确：
**图形仅用每5步诊断点，会遗漏Step41/47的梯度尖峰；判断尖峰必须看每步日志。**
热图灰色表示有效样本不足，不是零熵。

尚无0.6B benchmark得分。原队列是先完成200步训练，再对Step50/100/200
各自完整评测；不为提前出分中断训练。每个benchmark独立计分的规则不变。

## 夜间巡检：Token 恢复测试通过并开始正式训练

以下为北京时间2026-09-13的实际观测，不是预计启动时间。
用户要求直接监督实验；本轮采用只读SSH巡检，没有新增后台巡检服务，
没有重启队列、修改训练参数、覆盖runtime或连接train。

- 04:21:05，旧窗口恢复队列完整结束，`queue_state.json.status=complete`，
  `recoveries/jsonl_20260912_r1/protected_inputs_verified.json.passed=true`。
  Sliding3 Step100的四任务原始及内置评分JSONL均逐物理行解析通过，
  回答数依次为4000、240、240、664，退出码0；Step200及历史重评分随后结束。
- 04:21:31，0.6B队列开始新阶段，runtime仍为`ec0a7a9`，
  `budget_seconds=null`，没有抢占前驱。运行时和资产检查重新通过，
  两组正式run-card一致性检查写入`paired_preflight.json`。
- 04:23:12，Token probe1启动；04:35完成Step1保存及审计。
  04:36:22，probe2从该checkpoint恢复；04:47完成Step2及恢复门禁。
  两个checkpoint均包含四rank的model/optim/extra_state，共12个非空状态文件，
  另有`data.pt`及模型/tokenizer配置。四rank的加载日志、scheduler步数、
  RNG字段和dataloader消费位置均通过`probes/token_opd/resume_gate.json`检查。
  这不证明vLLM内部随机流能够逐位恢复。
- 04:47:25，正式Token训练进程PID **3594635** 启动，控制器仍为PID3547500。
  正式run-card为`resume_mode=disable`、空`resume_from_path`，从原始官方Base
  重新开始200步，不接续probe权重。约04:58完成正式Step1。

正式Step1诊断：裁剪前grad norm36.9889、student entropy0.0416895、
teacher entropy0.0353452、response mean8394.25、截断率0.5；
所有nonfinite/overflow计数为0。首批prompt SHA为
`00e902b1ea756ff9ec0cb6e86e8c1e0c17418ff6b448186026d0b9803ec5e402`，
与probe1一致。高梯度、低熵和长回答需要多批次跟踪，不能依据一个点
宣称爆炸、崩塌或长期稳定。Block3尚未训练，不能声称已验证方法有效。

两次probe在完成保存、输出最终指标之后，析构阶段各出现一次
`DataLoader worker ... killed by signal: Killed`，两作业和审计均exit0。
它们没有发生在正常训练循环内；根因尚未证实，不能直接等同于GPU OOM，
也不能把整个日志描述为“零异常”。当前容器无可读cgroup内存计数器，
`dmesg`访问被拒绝，不能声称已从内核日志排除host OOM。
截至04:59，正式训练未复现此告警；原始记录保留在两个probe作业日志中。

本轮重新验证全套测试：510 passed、2 skipped。只读核对两个冻结runtime
共1312/1274个文件及30个受保护模型、数据、grader文件，全部通过。
后续仍按Token完整训练/三轮完整eval后再Block3的既定顺序执行；
各benchmark单独报告，不以总分替代。

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
