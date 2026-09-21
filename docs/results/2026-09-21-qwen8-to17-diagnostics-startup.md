# Qwen8对1.7B两组能力验收启动记录

## 实际状态

2026-09-21 19:39:17北京时间在ml2启动独立后台队列，控制器PID `1872503`。
19:42核验 `PPID=1`、`SID=1872503`，不依赖本地Windows或SSH连接。
19:43巡检：1.7B-Base资产任务已完成，正在只读校验已有8B-Base资产；四张GPU空闲。
**截至该快照没有新模型能力得分，也没有启动OPD训练。**

- 运行根目录：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260921v1_qwen8_to17_diagnostics_ml2`。
- 控制代码：`ce653fbe5a9aebceadd37d6f8cfd33e46a1bd11d`，已推送内部GitHub分支 `feat/qwen4-blockfirst`。
- 不可变部署：上述仓库根目录下 `analysis_deployments/ce653fbe5a9aebceadd37d6f8cfd33e46a1bd11d`。
- 启动方式：在该部署目录执行 `nohup setsid env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 <verl-python> scripts/run_qwen17_pair_diagnostics.py`。
- Python：`/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl/bin/python`。
- 记录文件：运行根目录下 `controller.pid`、`controller.log`、`queue_manifest.json`、`queue_state.json`；各阶段日志在 `queue_jobs/<job>/logs/job.log`。
- 环境固定：torch2.8.0、transformers4.57.6、vLLM0.11.0、tokenizers0.22.2；独立缓存 `/dev/shm/q817d`。

## 模型与顺序

全部使用官方ModelScope模型，原始文件大小、SHA256和完整revision固定；不使用私人训练模型。

| 顺序 | 模型角色 | 官方模型 | ModelScope revision |
| --- | --- | --- | --- |
| 1 | Base学生 | Qwen/Qwen3-1.7B-Base | b0786a09cd6ee101cd8c90e30a5727beb8230544 |
| 1 | Base教师 | Qwen/Qwen3-8B-Base | 932bc907a0f908fd665867dec24af47c2f57e719 |
| 2 | 指令学生 | Qwen/Qwen3-1.7B | 4855588ea1a12789f2e965e5f52a9e4a24c94b2a |
| 2 | 指令教师 | Qwen/Qwen3-8B | 26028140be3ee69b82b1d1450179ab71bb1121b9 |

指令组的官方名称没有 `-Instruct` 后缀，两者是支持thinking开关的后训练模型。
使用原生聊天模板并显式设置 `enable_thinking=False`；每个模型先跑4条GPU检查。
输入模板中预填的空think块与输出中新生成的think标签分开记录，不能通过删除输出标签掩盖失败。
Base组仍使用completion数学提示，不保留ChatML。

## 对照和解释边界

实验协议见[冻结计划](../plans/2026-09-21-qwen8-to17-diagnostics.md)。

- 复用此前封存的64道DAPO诊断题，每个模型独立答题64题x2次。
- 每组使用自身原始1.7B学生sample0的中途前缀，对固定前32题由师生各续写2次。
- 两组共768条诊断轨迹，另加8条指令模型GPU检查；所有原始token、提示、输出、seed、结束原因和评分保留。
- 同组师生题目、提示、输入token、采样配置及预算一致：温度1、top_p0.9、top_k=-1，响应总预算16384token。
- 历史grader不变；各组独立报告正确率、配对题目bootstrap区间、截断、周期重复、无有效boxed率和新生成think标签。
- 预冻结教师门槛不变：独立答题至少+5个百分点且配对95%CI下界大于0，续写差值非负，并满足历史健康指标门槛。
- Base能力不合格或证据不足不阻止继续指令组；资产、版本、原始记录或GPU非thinking检查失败则停止，不自动重试或调参。
- 无论结果如何都不启动训练。此轮不是四个benchmark的完整评测，也不是OPD算法有效性的实验。
- 两组使用不同的原生提示形式和后训练模型，因此组间比较不是纯参数规模消融。
- 64题曾在上一轮8B对4B诊断中使用；不能宣称是本轮新留出的盲测题或模型预训练未见过的数据。

## 工程验证

- 独立代码复核发现的批内原始轨迹保留、停止原因校验和运行版本漂移问题已经修复，复核未发现剩余启动阻断问题。
- ml2实际verl环境执行资产、资格验收、控制器及历史block监督共 **264项CPU测试，全部通过，26.20秒**。
- 首次将pytest临时目录放在NFS上时262通过、2失败：两项安全测试在构造硬链接测试文件时被NFS拒绝，尚未进入被测逻辑。
  未降低安全断言；将pytest临时目录改为可执行本地tmpfs后，完整264项通过。
  正式资产发布不依赖硬链接，仍使用排他发布和完整哈希校验。
- 测试日志保留在仓库根目录下 `diagnostics/20260921_qwen17_pair_preflight/pytest.log` 和 `pytest_local_tmpfs.log`。
- 语法编译及 `git diff --check` 通过；旧训练runtime、旧8B对4B结果与模型文件未改写。
- 本文后续文档提交不替换运行中的不可变代码部署。
