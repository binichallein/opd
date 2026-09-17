# 0.6B Block3完整评测

用户明确要求直接启动评测，无需再询问。仅ml2，不启动新训练，不恢复旧队列。

## 固定协议

- 源为已完成的`20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2/block3_mean`。
  按200、100、50顺序完整评测，顺序与前次Token评测相同，不按分数跳过任何节点。
- 新评测目录`20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2`；
  编排脚本`scripts/run_nonthinking_block3_eval.py`。
- 评测器和merger固定在`0ce73aa42d7f734b9d34c379f474458bb6d48771`。
  新编排脚本仅选择Block3 checkpoint来源，并复用已验收评测/审计辅助函数。
- 四张A100，MATH500=500题、AIME24=30题、AIME25=30题、AMC23=83题。
  每题8次、seed21至28、temperature1、top_p0.9、max_tokens16384；显式关闭thinking，
  提示格式与Token评测一致。每个checkpoint5144条，合计15432条响应。
- 历史grader SHA256为
  `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
  不换grader、不用小样本、不添加新的截断过滤或重复惩罚。沿用评测vLLM预算0.9，
  不是训练侧0.6；不把该区别说成两组评测协议不同。

## 配对与产物保护

先复核旧队列已完成、Block3完整状态/轨迹已验收、两组run_card配对一致。
Token分数直接复用v1三个已验收视图，重新检查原始/判分行、数据身份、采样覆盖、
汇总与证据哈希，不重新生成Token输出。新HF合并和评测只写v2目录。

每个视图保存命令、PID、版本、模型哈希、实际提示token契约、原始输出、逐题
判分和验收证据；按benchmark分别报告Avg@8、Pass@8及同step差值，不合并总分。
旧模型、数据、Token与Block3 checkpoint等受保护文件在前后复核，不删产物。
遇到非零退出、覆盖不完整或哈希不符就停止，不自动重试/改参。

## 执行检查

1. [x] 只读检查ml2空闲、v1 complete、三个完整checkpoint存在。
2. [x] 新编排与来源选择测试先失败再通过；复用原评测及数据验收逻辑。
3. [ ] 完成代码复查、远端测试、冻结版本部署和nohup启动。
4. [ ] 确认首个checkpoint已完成合并和实际提示检查并进入四GPU生成。
5. [ ] 三个完整评测全部验收，输出逐benchmark对照报告。

现有训练输出退化是评测背景，不是提前判定分数或跳过评测的理由。
单seed点估计不能证明跨seed稳定性或退化的因果机制。
