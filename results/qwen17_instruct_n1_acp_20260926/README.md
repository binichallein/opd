# ACP 8B 到 1.7B 指令版：100步、32x1

此目录目前保存**启动验收证据**，不是 benchmark 结果。运行协议见
`docs/plans/2026-09-26-acp-qwen17-n1.md`。

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
