# Window OPD 评测恢复记录

## 故障与证据

2026-09-12 19:44（北京时间），两组训练均完成后，Random3 的三轮生成与
内置评分正常退出 0，但 full audit 因 `JSONDecodeError` 退出 1，队列按协议停止。
Sliding3 评测尚未开始。失败目录、原始队列日志及退出码全部保留。

根因是审计脚本对 JSONL 使用 `str.splitlines()`，会将 JSON 字符串中合法的
U+0085、U+2028、U+2029 误认为记录边界。同一错误也存在于重评分和最终比较
的 JSONL reader。Random3 Step100 的 MATH500 第 3935 个物理行含 U+2028，
能稳定复现原异常；Step200 也包含这类字符。

已按物理行检查三轮 raw/graded 文件：每轮均有 4,000 + 240 + 240 + 664
条记录，合计 15,432 条原始回答；全部 JSON 可解析，检查的配对键无重复。
不是模型 OOM，不需要重新训练、重新采样或改写回答来绕过问题。

## 修复边界

- 三个分析入口改为按物理行读取，不改变答案提取、符号判等或评分规则。
- 回归测试覆盖三种 Unicode 分隔符、LF/CRLF、源文件字节不变，以及真正
  截断的 JSON 仍报错。修复前 15 个相关用例失败，修复后通过。
- 全量本地测试：482 passed、2 skipped；跳过项为本地缺失绘图库的测试。
- 训练 runtime 仍为 `fbad852a638de18e20d571a061b2be0437942a38`。
  修改仅部署到独立的 `analysis_deployments/<analysis_commit>`，不覆盖该 runtime。
- 历史 grader 仍为 SHA-256
  `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
  内置 VERL 分数继续保留，不择优选择评分器。

## 恢复合同

新增 `scripts/recover_window_evaluation.py`，仅支持本次 ml2 队列：

1. 持有原 `queue.lock`，确认两组训练完成且 checkpoint/window 审计通过。
2. 仅接受已人工检查的 Random3 full-audit 失败；拒绝既有 Sliding3 评测目录，
   避免覆盖或隐式重试。恢复 ID 必须唯一。
3. 保存原队列状态、日志、PID、旧 acceptance 副本，并记录已有 raw/graded
   结果、诊断及元数据的保护哈希。
4. 使用新分析代码重新审计 Random3，再对 Step50/100/200 运行固定历史 grader。
5. 使用原冻结代码生成 Sliding3 的三轮完整四任务 n=8 结果，再统一重评分。
6. 执行原定的题目分层配对 bootstrap，主终点仍为 Step200，最后再次验证
   保护哈希。任何新失败都停止并保留产物，不自动改参或重试。

恢复审计和评分日志位于 `recoveries/<recovery_id>/`，原失败的
`random3/queue_jobs/full_audit/` 不改动。生成与模型合并仍使用冻结代码，
分析 commit 与训练 commit 分开记录。恢复启动证据和结果将在实测后补充。

## 用户新目标与当前权限

用户要求修复、按历史 grader 评测两组，并根据证据寻找 Block3 以外的增益
方法，或检验 Block3 的跨师生/跨数据泛化，最终形成可写论文的实验材料。
这不意味着可以保证正结果。先完成当前两组，再形成有成功/失败门槛、
成本上限和反例保留规则的下一阶段方案；没有自动扩大 training seed、
模型对或数据集的预算。本次仍不连接 train。
