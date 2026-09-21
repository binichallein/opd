# Qwen8 教师验收巡检快照

## 最终补记

19:06实时复核：原控制器已于17:36正常结束，状态`capability_not_accepted`，
`training_started=false`。旧GRPO续写最终16/64正确（25.00%），截断1/64，
周期重复0/64，无有效boxed1/64，新生成think标签0。参考续写与最终汇总均exit0。
最终gate为`inconclusive`，仅`direct_gain`、`direct_ci`失败，其余检查通过。
报告SHA256与sidecar一致：`ef60b0889ff0041432ddea91c59bf9bf5641dde716f3df62081bd8bd9bc396f8`。
四GPU空闲；无训练/探针目录。以下保留17:25快照，不改写当时的待完成状态。

## 范围与状态

2026-09-21 北京时间17:11开始现场巡检，17:13:44至17:23:46另做每30秒一次的只读采样。
本快照记录到17:25附近的核验结果，不是最终六单元验收报告。
未修改代码、采样设置、训练参数或验收门槛，未启动额外GPU任务。

实验根目录：
`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260921v1_qwen4_from8b_completion_seed21_ml2`。
控制release仍为 `55e24cfc2a4a3891d348a8247b712e4ead4e0f6a`，控制器PID `1839519`。

- 三个模型的独立答题均已完成，学生与8B的续写也已完成。
- 17:23:10自动进入旧4B-GRPO教师续写，子任务PID `1852491`。
- 此时无 `probes/`、`token_opd/` 或 `block3_mean/` 训练产物目录，尚未发生训练更新。
- 必须读取实时 `queue_state.json` 和最终 `teacher_qualification/gate_acceptance.json` 后，
  才能报告最终队列状态。本快照不能代替最终报告。

## 已完成结果

这是固定DAPO诊断题，不是MATH500/AIME/AMC完整benchmark评测。
独立答题64题、续写32题，每题2次；正确率是回答级平均正确率，不是Pass@2。

| 阶段 | 模型 | 正确数/回答数 | 平均正确率 | length-stop | 周期重复 | 无有效boxed | think标签 |
|---|---|---:|---:|---:|---:|---:|---:|
| 独立答题 | 原始Qwen3-4B-Base学生 | 16/128 | 12.50% | 3/128 | 3/128 | 7/128 | 0 |
| 独立答题 | Qwen3-8B-Base候选教师 | 13/128 | 10.16% | 2/128 | 1/128 | 4/128 | 0 |
| 独立答题 | 旧Qwen3-4B-Base-GRPO参考 | 42/128 | 32.81% | 5/128 | 1/128 | 6/128 | 0 |
| 学生前缀续写 | 原始Qwen3-4B-Base学生 | 11/64 | 17.19% | 1/64 | 1/64 | 0/64 | 0 |
| 学生前缀续写 | Qwen3-8B-Base候选教师 | 12/64 | 18.75% | 1/64 | 0/64 | 1/64 | 0 |
| 学生前缀续写 | 旧Qwen3-4B-Base-GRPO参考 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |

按题目配对bootstrap，10,000次重采样，seed21，使用冻结实现计算：

- 8B减学生，独立答题：-2.34375个百分点，95%区间[-9.375, +4.6875]个百分点。
- 8B减学生，续写：+1.5625个百分点，95%区间[-4.6875, +7.8125]个百分点。
- 旧GRPO减学生，独立答题：+20.3125个百分点，95%区间[+10.9375, +29.6875]个百分点。

8B独立答题不满足预先冻结的“至少+5个百分点且95%下界大于0”必要条件。
因此按当前记录不具备自动启动训练的资格。续写的微小正差不能抵消独立答题门槛失败。
这不证明8B在所有任务上弱于4B，也不证明它在任何OPD配置下都无效。
最终六单元重验及状态写入仍由原控制器完成，禁止调阈值或追加样本直到通过。

## 工程检查

- 已完成五单元的原始轨迹、评分、请求与manifest都通过各自completion记录的SHA256校验。
- 三个独立答题单元的协议、请求hash、选题hash、历史grader hash、软件版本完全一致。
- 对三个独立答题及两个续写的请求JSON做了逐项相等比较：输入token、前缀、seed和预算一致。
- 学生、8B续写均使用同一原始学生轨迹hash
  `6b9baf7334aff626adcf0181a859f2ae4cf03de1ecbd59025a602aae94b0383b`。
- 历史grader保持
  `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
- 所有已完成子任务exit code均为0。轮询日志未发现OOM、EngineDeadError、
  FloatingPointError或Traceback；未据此声称未检查的后续阶段安全。
- 8B续写GPU0显存稳定在约48,335MiB，其他GPU空闲；切换模型时GPU释放。
- 8B最后一条续写前缀788 token、剩余预算15,596 token，在17:23左右以length结束。
  长时间停留63/64来自最后一条长输出，完成计数不是逐token进度。
- vLLM退出时有未显式destroy_process_group的NCCL警告，但子进程exit0、GPU释放、
  completion写出并通过hash校验，队列已正常进入下一单元。没有因此重启或改参数。

## 证据位置

全部路径相对于实验根目录：

- `teacher_qualification/{direct,continuation}/{student,teacher,reference}/`
  下的 `raw.jsonl`、`results.jsonl`、`requests.json`、`manifest.json`、`completion.json`。
- `queue_jobs/qualify_*/logs/job.log` 及 `exit_code.txt`。
- `controller.log`、`queue_state.json`、`queue_manifest.json`、`protected_inputs.json`。
- 最终报告预期为 `teacher_qualification/gate_acceptance.json`，本快照时尚未生成。

本次前台只读巡检已结束；原服务端验收队列继续运行，不代表助手离线后仍在实时巡检。
