# Qwen8教师资格验收队列启动

## 当前事实

2026-09-21 16:14:23北京时间启动独立后台控制器，PID `1839519`，
已核验 `PPID=1`、`SID=1839519`，不依赖本地Windows或SSH会话。
16:15巡检时仍处于教师资产下载阶段，首个权重分片已写入约935MiB。
**教师能力尚未验收，正式训练尚未开始，没有新增benchmark分数。**

- 执行机器：仅ml2，4张A10080GB。下载不占GPU，巡检时GPU均空闲。
- 教师：官方ModelScope `Qwen/Qwen3-8B-Base`，revision
  `932bc907a0f908fd665867dec24af47c2f57e719`。
- 学生：同一原始官方 `Qwen/Qwen3-4B-Base`，不从旧Token/Block3权重续训。
- 控制代码：`55e24cfc2a4a3891d348a8247b712e4ead4e0f6a`；
  训练代码仍冻结在 `0f9161f02f08287fb07f0375ad0a6bda81133ff0`。
- 新根目录：
  `/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260921v1_qwen4_from8b_completion_seed21_ml2`。
- 命令：在对应immutable analysis deployment内通过服务端
  `nohup setsid env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 <verl-python> scripts/run_qwen8_teacher_pair.py`
  启动，stdout/stderr记录到新根目录的 `controller.log`。
- 状态：`queue_state.json`；配置：`queue_manifest.json`；控制器PID：`controller.pid`。
  资产子任务PID `1839598`，日志 `queue_jobs/teacher_assets/logs/job.log`。

## 顺序和通过条件

完整标准已在任何新模型生成前固定于
[验收与对照计划](../plans/2026-09-21-qwen8-teacher-acceptance.md)。

1. 下载并检查全部15个原始模型文件，记录来源、大小、SHA256、tokenizer和EOS。
2. 固定64道DAPO诊断题，每题2条，分别测试原始学生、8B候选、旧4B-GRPO参考。
3. 固定前32道题，在同一学生sample0中途前缀上，三模型各继续生成2条。
4. 验收要求8B独立正确率相对学生至少+5个百分点且配对题目bootstrap95%下界>0；
   续写差值非负，并通过截断、周期重复、有效boxed答案、原始轨迹完整性检查。
5. 没通过或证据不足则结束队列，不调阈值，不自动扩样本，不启动训练。
6. 通过后两方法各做Step1保存、退出、恢复到Step2；检查四rank优化器、调度器、
   RNG、dataloader状态、真实rollout、全部诊断与输入seed合同。
7. 两门禁通过后从原始学生分别训练Token200、Block3Mean200；保留50/100/150/200
   所有完整状态、全部6400条rollout/组、200步诊断及位置热图。训练算法、数据和seed不变。

此次控制器不自动追加完整benchmark评测。64题资格检查不是MATH500/AIME/AMC的完整评测。
后续完整评测仍需使用历史grader、n8，四任务分别报告Avg@8/Pass@8。

## 数据预检

同一1791700行DAPO池保留不变。首17917行在空白归一化后有12组同题不同答案，
只在验收选题中排除整个歧义组，同时排除四任务重合题和历史seed21/200步实际训练题。
排除清单、原始来源、选题SHA和采样参数全部保留；不声称这些题对教师预训练或历史RL未见过。

## 工程修复与验证

- 首次CPU资产准备release `33d883d`、PID `1837814` 在元数据发布时遇到NFS硬链接
  `EPERM`，未下载权重、未运行GPU任务。没有对同一个失败正式任务自动重试。
- 失败目录的两个文件完整移至
  `diagnostics/20260921_qwen8_teacher_assets/failed_init_33d883d`，原下载日志仍保留。
- 新发布逻辑优先 `renameat2(RENAME_NOREPLACE)`；不支持时排他创建目标并校验复制结果。
  不覆盖已有文件，不放松大小或哈希校验。实际NFS路径上9个专项测试通过。
- 原拟定临时目录超出Ray socket107字节上限，改用独立 `/dev/shm/q48`；
  两组路径的回归测试通过。正式持久产物仍在原实验存储，不放tmpfs。
- ml2实际verl环境CPU验证：**183项测试通过**，包含资产、资格检查、控制器、原始
  block监督和历史4B控制逻辑。资格生成/计分的单测包含六单元完整模拟与原始产物篡改拒绝。
- 9项NFS专项测试中的pytest缓存写入警告不影响测试退出码；后续全套测试关闭pytest缓存。
- 两个新提交已推送内部GitHub分支 `feat/qwen4-blockfirst`；此后文档提交不替换运行中release。

## 解释边界

8B必须先证明在本设置下能指导4B学生，不能仅凭参数量认定合格。
与旧4B-GRPO教师比较时同时改变规模和后训练经历，不属于纯教师规模消融。
Block3仍包含历史advantage聚合、联合ratio和归一化，不能把此对照称为纯advantage平滑。
