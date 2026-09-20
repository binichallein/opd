# Qwen4 Token完整评测启动

北京时间2026-09-21 00:55:40，ml2独立后台控制器1783812启动，PPID1且SID1783812。
控制发布`eb0198e596df91d49a5570cc6a406e02dc460638`，训练/推理冻结发布仍为
`0f9161f02f08287fb07f0375ad0a6bda81133ff0`。命令入口：
`scripts/run_qwen4_completion_eval.py --variant token_opd`。

目录：`runs/20260921v1_qwen4_completion_token_eval_seed21_ml2`。
顺序Step200、150、100、50，每权重完整四benchmark、5144轨迹。
00:56:21完成运行时hash核验后进入Step200 CPU合并。此记录是启动证据，不代表评测完成；
读取远端queue_state/evaluation_acceptance及原始输出后再报告结果。

训练于00:26:07退出0，00:29:02完成后验收。四完整权重和200步诊断/轨迹保留。
200步两组共同比较输入核验通过，每组6400条，Token训练累计cap263/6400(4.1094%)，
Block3为247/6400(3.8594%)。这些是训练cap指标，不是benchmark引擎截断率。

本地27项相关测试通过、diff检查通过，独立代码审查未发现阻塞问题。
新评测部署归档SHA256为`0be0d5ddbbb216eadc4d1dacb028f47321a686df83265cdd6f5664a1b61681db`，
本地与远端控制器SHA256均为`1ef37e6530740f5d763a4e5c6df286c2367834fcbe34e7e3de4f4836c2c8d7d1`。
未修改任何历史部署、训练参数、权重或已完成Block3评测。
