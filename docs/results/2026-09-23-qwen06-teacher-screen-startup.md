# 0.6B八组教师能力筛选：启动记录

## 范围与状态

用户批准八组配对，详见[冻结方案](../plans/2026-09-23-qwen06-teacher-screen.md)。
本轮仅在ml2运行推理，不训练、不进行四benchmark完整评测，也不替换历史运行代码。
主验收使用学生的实际输入和生成前缀；教师原生模板独立答题另作对照。

2026-09-23 00:35:55北京时间，服务端后台队列启动。
控制器PID `2648547`，启动后核验PPID为1，SSH断开不会终止该队列。
启动阶段先下载缺失的官方 `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-4B`。
官方名称没有`-Instruct`后缀；两者的ModelScope完整revision及文件SHA256已冻结。
其余五个模型在现场完成只读哈希和原生提示渲染检查。
公共GRPO模型的config、generation_config和tokenizer原生EOS均为151643，未修改。

此文记录启动事实，**不是完成或教师有效性的声明**。动态状态以运行目录为准。

## 可复核入口

- 运行代码commit：`ad990c91f62f1e41e191b48ee7f9a0994046465c`。
- 不可变代码：`opd/analysis_deployments/ad990c91f62f1e41e191b48ee7f9a0994046465c`。
- 运行目录：`opd/runs/20260923v1_qwen06_teacher_screen_ml2`。
- `opd`绝对根目录：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
- `queue_manifest.json`记录授权范围；`queue_state.json`记录阶段和GPU任务。
- `jobs/*/command.json`、`started.json`、`finished.json`和`output.log`保留每项命令与进程证据。
- `qualification/prepare_manifest.json`固定数据、模型、实际请求、原生对照引用和环境。
- `qualification/cells/*`保留原始token、文本、采样logprob、停止原因、历史grader结果及封存哈希。
- 完成后输出`qualification/pair_summary.json`和`pair_summary.csv`，分对、分阶段统计。

## 对照约束

复用历史64道DAPO诊断题。独立解题每题2次，前32题取学生sample0的固定中途前缀，
师生各续写2次。学生两种原始模型各只生成一次基线，所有对应教师共用。
原生教师对照只有在实际输入token及采样完全一致时才复用已有单元。

主比较不对教师重新套模板，也不重新编码学生前缀。所有模型的显式stop集合相同，
为`[151643,151645]`，同时保留模型自身原生EOS；这是新版本，不冒充旧Base诊断的逐位复现。
生成总预算16384 token，续写扣除已有前缀；temperature1、top_p0.9、top_k=-1，
沿用按题目/样本/阶段派生的请求seed。四卡TP1独立队列不会把不同模型挤到同一卡。

CPU验收：新增14项测试及相关旧回归合计161项通过；控制器CPU导入检查通过。
GPU检查和正式推理的结果需要由实际运行产物确认，CPU测试不能代替。

## 解释边界

每对分别报告独立答题增益、续写增益、配对题目bootstrap95%区间、截断、周期重复、
缺有效boxed和生成think标签；教师原生输入与学生输入之间的差距另列。
这些是诊断正确率，不是历史完整评测的Avg@8或Pass@8。

公共GRPO教师自身用DAPO训练过。这批题也已用于旧诊断，不能宣称新盲测或教师未见数据。
八对共享题目和基线，未做多重比较校正；筛选通过不等于已证明OPD能涨点。
若把筛选作为论文贡献，仍需用独立师生与实际蒸馏收益检验预测性。
