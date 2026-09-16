# 0.6B Token 完整评测与 Block3 顺序队列启动记录

本文记录2026-09-17的启动快照，不代替实时状态，也不是准确率结果。
完整方案见[评测与训练方案](../plans/2026-09-17-qwen06-token-eval-then-block3.md)。

## 运行身份

- 仅ml2；根目录为`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
- 新运行：`runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2`。
- 控制器冻结版本：`788a2dca21902fda25777fca1a6863780973a8c1`。
- 训练及评测冻结版本：`0ce73aa42d7f734b9d34c379f474458bb6d48771`，与已完成Token相同。
- 控制器PID4130269，2026-09-17 02:21:24北京时间启动，已脱离SSH会话。
- Token Step200评测PID4130750，02:23:41启动。完整命令保存在其评测目录。
- 实时入口：新运行的`queue_state.json`、`controller.log`和各任务的日志。
  不根据本文快照重复启动控制器，不恢复旧thinking队列。

## 已完成检查

- 新代码完整测试：535 passed、2 skipped；独立代码复查未发现阻塞项。
  测试和静态复查不等于远端完整评测、恢复测试或正式训练已经完成。
- 两份冻结runtime文件哈希通过；旧Token已完成200步，50/100/200完整状态
  checkpoint均保留。已记录旧checkpoint及验收文件哈希，后续再次校验。
- Step200 FSDP权重已合并到新运行的`merged/token_step200`。原checkpoint
  中配置/tokenizer位于`actor/`根目录；冻结merger支持该布局，没有修改旧文件。
- `paired_preflight.json`为passed：两组模型、输入数据哈希、runtime、seed21、
  lr2e-6、200步、micro-batch1、训练vLLM占比0.6、解码、prompt协议及观测配置一致。
  方法差异仅为k1/sum改为固定非重叠k3/mean，另有实验名和新输出目录等身份差异。
- Step200的643道题实际token IDs与官方Base按训练协议渲染的IDs一致，最长818。
  显式`enable_thinking=False`，空且已关闭的think前缀保留；没有触发2048输入边界。
- 历史grader哈希与正反例自检通过：
  `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
- 已启动Step200的MATH500四GPU评测工作进程；初始快照仍在vLLM引擎初始化。
  当时尚无完整benchmark分数，Block3的GPU恢复测试及正式训练均未开始。
- 02:26:39前四个vLLM引擎均完成初始化；随后实测四卡GPU利用率分别为
  81%、82%、84%、76%，显存各78953MiB，已进入生成阶段，未见启动异常或OOM。
  此处评测独占GPU的显存预算沿用冻结评测器默认0.9，不是训练侧0.6的变更。
  评测器关闭逐响应进度条并在任务完成后写JSONL，暂未落盘不等于没有生成。

## 顺序与计分

评测顺序为Token Step200、100、50、官方Qwen3-0.6B-Base、现有公共
Qwen3-4B-Base-GRPO教师。每个视图均完整覆盖MATH500 500题、AIME24 30题、
AIME25 30题、AMC23 83题，每题8次生成，seeds21至28，temperature1、top_p0.9、
最大输出16384。共5个模型视图、25720条响应，不用小样本分数替代完整评测。

每个benchmark分别报告Avg@8、Pass@8及Token相对未训练学生的百分点变化，
不合并总分。单seed点估计不代表跨seed显著性，教师分数也不自动构成学生上界。

## 后续自动执行与保护

只有五个视图全部完成，且原始输出、题目身份、8次采样覆盖、逐题判分与
汇总分数一致性检查通过，才生成`token_effect.json`、`token_effect.md`和
`token_eval_acceptance.json`，并进入Block3 GPU阶段。分数好坏不是跳过对照的条件。

Block3先做两步完整保存/恢复测试并审计四rank状态、数据进度及原始轨迹，
然后正式从官方Base训练200步，绝不接着Token或探针权重训练。每步保存完整
rollout，Step1及每5步保存原有标量与位置诊断，50/100/200完整checkpoint全部保留，
结束后绘制诊断曲线和热图。本控制器不自动启动Block3的完整benchmark评测。

非零退出、数据不完整、哈希变化或配对异常会停止队列，不自动覆盖、重试或
删除产物。所有新合并、评测、探针及Block3文件写入新运行目录，旧Token只读。

## 产物入口

- `evaluations/<模型视图>/eval_card.json`、`prompt_contract.json`及`model_identity.json`。
- `evaluations/<模型视图>/logs/eval.log`和`outputs/`中的原始JSONL、逐题判分及summary。
- `evaluations/<模型视图>/acceptance.json`：完成后的独立覆盖及计分验收。
- `paired_preflight.json`、`protected_inputs.json`：配对配置和旧输入保护记录。
- `block_launch_gate.json`：评测完成后允许进入Block3 GPU阶段的记录。
- `block3_mean/rollouts/formal/step_XXXXXX/raw.jsonl.gz`及哈希：未来正式训练原始轨迹。
- `block3_mean/diagnostics/`、`checkpoints/`和`figures/`：未来正式观测、完整状态和图表。
