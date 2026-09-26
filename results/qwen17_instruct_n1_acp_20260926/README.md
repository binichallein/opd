# ACP 8B 到 1.7B 指令版：100步、32x1

此目录保存启动验收证据及**最终完整评测结果**。2026-09-26北京时间12:02:12，
两组100步训练与8个权重的完整评测全部完成。运行协议见
`docs/plans/2026-09-26-acp-qwen17-n1.md`。

## 最终结果

- `evaluation_acceptance.json`：8个权重均完成每题8次的四benchmark评测并通过验收。
- `per_benchmark_results.json`：各checkpoint的Avg@8、Pass@8、格式错误及截断率。
- `paired_comparison.json`：配对差值，不合并benchmark计分。
- `paired_rollout_acceptance.json`：100步题目顺序与输入配置的配对一致性，两臂各3200条轨迹。
- `block3_mean/`、`token_opd/`：完整状态和轨迹验收、run card、过程曲线与位置热图。
- 原始轨迹和所有完整checkpoint仍保存在AFS，不将大权重提交Git。

主报告点Step100，以下均为百分比：

| Benchmark | Token Avg@8 | Block3 Avg@8 | Token Pass@8 | Block3 Pass@8 |
| --- | ---: | ---: | ---: | ---: |
| MATH500 | 72.60 | 71.98 | 90.40 | 90.80 |
| AIME24 | 14.17 | 15.42 | 30.00 | 30.00 |
| AIME25 | 11.67 | 12.08 | 33.33 | 23.33 |
| AMC23 | 43.98 | 42.47 | 77.11 | 72.29 |

这一师生对的结果有升有降，不能声称Block3在所有任务、指标或checkpoint上均优于Token。
这里只完成一个训练seed，未据此声称统计显著性。其余25/50/75结果保留在完整JSON中。

## 启动证据

- 运行代码固定为 `1867093b21799c36be22bb2fd167c87e5aaa039e`，后续文档提交不替换运行代码。
- 服务端根目录：`/mnt/afs/202609/tyf-qwen-opd/runs/20260926v1_qwen8_to17_instruct_n1_step100_seed21_acp`。
- `preparation/` 的 JSON 是从服务端复制的时间快照；当前进度必须读取服务端 `queue_state.json`。
- `final_environment.json` 是补齐 FlashInfer 后的版本记录，以此为准。
- `block3_probe1_*` 仅证明保存测试的完整状态和实际轨迹检查通过，不证明从该状态恢复成功。
- `block3_probe2_*`、`block3_resume_gate.json` 另行证明从Step1恢复、更新并保存Step2成功，
  四rank状态和64条轨迹均通过审计；不承诺vLLM逐位重放。
- Probe 不进入正式训练曲线；正式两臂分别从原始学生初始化，不从 probe 或对方权重继续训练。
- 数据划分、采样位置、prompt 与 scorer 检查不等于 benchmark 性能提升证据。
- H100/CPU配额及当前32x1采样配置不同于历史4x8指令版实验，不能把跨实验差异全部归因于算法。
