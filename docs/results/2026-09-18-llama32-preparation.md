# Llama 3.2 1B-Instruct / 3B-Instruct 准备记录

## 范围和协议

用户于2026-09-18选择这对同代 Instruct 师生，通过 ModelScope 获取；复用原始
DAPO 数据、算法和四个 benchmark，不使用私有模型。完整配置见
[执行计划](../plans/2026-09-18-llama32-paired-validation.md)。

顺序：前序 Qwen06 Block3 三步完整评测完成 → 原始1B-Instruct零步评测 →
Token恢复门禁/正式200步/200、100、50完整评测 → Block3 mean恢复门禁/从
相同原始1B独立训练200步/200、100、50完整评测。不是每条训练只评最后一步。

## 已完成的准备

- ModelScope 原始 BF16 资产26文件通过大小与SHA256校验。学生revision
  `d3e551343d4d81508a0d226656b826c217e463cd`；教师revision
  `4e7231b81c151c73632184994ac9a0149fcb22fd`。
- 权重位于 ml2资产根目录下 `models/Llama-3.2-1B-Instruct` 和
  `models/Llama-3.2-3B-Instruct`；每份包含source_metadata、SOURCE_REVISION、asset_manifest。
- 学生/教师全部643题输入一致，最长806tokens。输入SHA256均为
  `73ffa2e8f92f48f643d149bb421754939ed03a6e97d3264205964feb9f9c917f`。
- 实际 MathEnvironmentManager / TrajectoryCollector 的16条CPU输入与评测
  IDs匹配，原生pad使用已有EOS128009，不增加词表token、不改初始化权重。
- 模板日期固定为18 Sep 2026；Llama评测显式输入IDs，避免重复BOS；原生停止符
  128001/128008/128009。评测保存实际生成token和finish reason，format与截断独立统计。
- 两臂PREPARE_ONLY生成命令/run_card并验证匹配；没有执行任何Llama训练或GPU推理。
- 初版代码 `6da4959`：ml2屏蔽GPU的回归测试500通过、2跳过；本地Git相关45通过、
  supervisor11通过，共556通过、2跳过。本地张量测试受既有 `iJIT_NotifyEvent`
  PyTorch依赖错误影响，已用ml2原环境CPU测试覆盖，不修改训练环境。
- 旧supervisor测试假定无实际评测进程，在ml2会等待；停止的是测试子进程，随后该
  文件在本地11项通过。当前正式Qwen评测未被中断。
- 对初版独立代码复查未发现启动阻断问题；GPU兼容性与四rank断点恢复仍须在前驱完成后验证。

## 资产目录与安全边界

资产根目录：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
CPU准备证据：`asset_preparations/20260918_llama32/`。
计划运行目录：`runs/20260918v1_llama32_1b_3b_nonthinking_seed21_ml2`。
前驱：`runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2`。

独立不可变部署不会修改 `deployments/current`、旧runtime、模型、checkpoint或结果。
队列失败即停，不自动重试/删文件；两臂均保存50/100/200完整状态与全部6400条名义
训练轨迹、41个诊断时点及热图。评测完整覆盖每模型643题/5144条回答，各任务分别计分。

本记录的准备成功不等于GPU门禁通过，也不代表这对师生已产生新实验得分。
