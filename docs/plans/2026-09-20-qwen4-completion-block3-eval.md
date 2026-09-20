# Qwen4 Completion Block3 完整评测

用户授权立即评测，从 Step200 开始。队列依次评测 Step200、150、100、50；
不自动启动原始学生评测或 Token 训练。本次只新增独立评测队列，不修改已完成训练。

## 固定协议

- 训练来源：`runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2/block3_mean`。
- 四个 checkpoint 均保留完整训练状态；CPU 合并 FSDP 分片到评测目录，原件不变。
- 沿用冻结执行版本 `0f9161f02f08287fb07f0375ad0a6bda81133ff0` 的 merger 和 vLLM evaluator。
- 提示协议必须显式传 `--prompt-protocol qwen3_completion_boxed_v1`，不使用 ChatML；
  thinking=False，裸题目加 step-by-step/boxed 指令与 Solution 结尾，与本次训练输入一致。
- EOS151643，不增加 ChatML stop token。CPU 对齐全部 643 道输入，完成后核验真实 GPU 输入。
- MATH500 500题、AIME24 30题、AIME25 30题、AMC23 83题；每题8次，合计5144条/权重。
- seed21-28，temperature1.0、top-p0.9、max_tokens16384，四张 A100；
  复用历史独立 TP1 evaluator，vLLM memory0.9（训练的0.6不是历史评测配置）。
- 历史 external grader SHA256：`04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
- 评测数据逐文件对照训练 run 的 data_manifest SHA256，禁止另取数据或改 grader。
- 保存所有生成文本、原始 token、实际输入、seed、终止原因和评分；各 benchmark 单独汇报
  Avg@8、Pass@8、缺 boxed 比例、实际截断率。旧 evaluator 附带的 macro 字段不作为报告结果。

## 执行与验收

- 新目录：`runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2`。
- 控制器：`scripts/run_qwen4_completion_eval.py`，单独 immutable analysis deployment；
  服务端 nohup 脱离本地会话。临时和编译缓存使用 `/dev/shm/opd-q4-e1`。
- 每个权重：校验数据和grader -> 合并 -> 核验模型/输入 -> 完整生成评分 ->
  检查题数、seed覆盖、原始轨迹一致性、真实 completion 输入和stop设置 -> 记录分 benchmark 结果。
- 完成当前权重的所有任务并通过验收才进入下一个。失败则停队列，保留已有原件和部分结果，
  不自动覆盖、重试、改参数或删除 checkpoint。
- 原训练队列已结束，状态不改写；查询评测应读取上述新目录的 queue_state.json。
