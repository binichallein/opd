# Qwen3-4B-GRPO到1.7B-Base教师能力测试 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 仅测试公开4B-GRPO教师对官方1.7B-Base学生的能力和输入适配，不启动或排队训练。

**Architecture:** 复用八组教师筛选的推理、归档、历史评分和配对bootstrap。新增独立配对入口，
只将模型集合、Base输入停止配置和输出目录参数化；保持原八组测试默认行为不变。
使用服务端nohup、独立不可变部署和新目录，任何失败保留现场，不自动改参重试。

**Tech Stack:** Python、vLLM 0.11.0、PyTorch 2.8.0、Transformers 4.57.6、ml2四张A100 80GB。

## 范围与固定协议

用户最新授权是“先只做教师能力测试，训练另定”。即使教师通过，队列也只结束并报告，
没有训练入口、权重更新或完整benchmark评测任务。只访问ml2，不访问train。

- 学生：官方`Qwen/Qwen3-1.7B-Base`，ModelScope revision
  `b0786a09cd6ee101cd8c90e30a5727beb8230544`。
- 教师：之前使用的公开`lllyx/Qwen3-4B-Base-GRPO`，复用历史资产哈希及本地下载revision证据。
  不是Qwen官方GRPO版本，也不使用用户私有模型。本轮只读验证已有模型，不下载或覆盖。
- 学生主输入采用现有`qwen3_completion_boxed_v1`：裸题目、boxed指令、`Solution:`后缀。
  不调用聊天模板，不插入think标签；允许普通文本逐步推理。
- 主比较教师逐token接收学生实际prompt及原始续写前缀，不重新套模板或重编码前缀。
- 为与现有Base训推协议一致，主比较不额外添加stop-token IDs，仅使用模型原生EOS151643。
  教师原生chat对照显式`enable_thinking=False`，保留空闭合think输入，停止集合[151645,151643]。
  对照因此是原生输入/停止策略的整体适配对照，不将差异单独归因于模板。
- 使用此前冻结的64道DAPO诊断题和同一按题目/样本/阶段派生seed。不是新独立测试集；
  GRPO教师用过DAPO，不能把本轮作为泛化性能结论。
- 独立答题：学生及教师学生输入视图各64题x2；续写：前32题各取学生sample0前缀，师生各续写2次；
  教师原生对照64题x2。总计512条正式输出，另两模型各4条256-token短检查，共520条。
- 温度1、top_p=0.9、top_k=-1，总响应预算16384，续写扣除已有前缀长度。
- 使用冻结历史grader，保存请求、输入token IDs、原始输出、停止原因、评分和哈希。
- 报告独立/续写正确率、教师减学生差值及逐题bootstrap区间（10000次、seed21），
  同时报告双方截断、周期重复、有效boxed缺失和生成thinking标记。短检查的截断不当作16K截断率。
- 沿用既有筛选门槛：独立增益至少5个百分点且95%区间下界大于0，续写增益不为负；
  教师截断率不超过10%且比学生高不超过5个百分点，重复不超过5%，缺有效boxed不超过25%
  且比学生高不超过5个百分点，双方不得生成thinking标记。
  通过不代表已证明OPD涨点；未通过不代表教师普遍无效。

## Task 1: 独立入口与回归测试

Files: 新增`scripts/qualify_qwen17_base_grpo.py`、`scripts/run_qwen17_base_grpo_screen.py`、
`tests/test_qwen17_base_grpo_screen.py`；小范围修改现有筛选器的学生集合常量、队列脚本选择常量。

1. 先写失败测试，验证仅一对正确模型、Base prompt与停止规则、原始前缀和seed保持、原生对照分离。
2. `uv run --no-project --python 3.11 --with pytest python -m pytest tests/test_qwen17_base_grpo_screen.py -q`应先失败。
3. 用私有模块实例复用旧实现，不修改历史模块全局状态；新增参数保持旧默认值。
4. 新旧筛选、资产、队列测试均须通过；校验只有推理prepare/generate/summarize命令，没有训练命令。

## Task 2: 部署与启动

1. 提交计划、代码与测试到Git；生成新`analysis_deployments/<commit>`及完整文件哈希清单。
2. ml2核实旧队列complete、四卡空闲、磁盘充足和模型哈希；运行相同CPU回归。
3. 新目录`runs/20260923v3_qwen17_base_grpo_teacher_screen_ml2`，新缓存`/dev/shm/q17gs0923`。
4. 服务端nohup启动，保存PID、PPID、SID、环境、完整命令和资源信息。退出SSH不终止测试。
5. 核实准备成功、实际GPU推理启动、首批原始响应落盘；不将“已启动”写作“已通过”。

## Task 3: 归档与状态

记录启动日志及证据到`docs/results/2026-09-23-qwen17-base-grpo-screen-startup.md`并备份本地，
Git只保存小型元数据及汇总，不上传权重/大批原始轨迹。完整测试结束后根据封存汇总给出验收结果。
本次队列在测试结束后停止，任何训练需要用户另行授权。

## 执行状态

- [x] 最新用户范围确认：仅能力测试。
- [x] 模型存在、旧队列complete、四张GPU空闲。
- [x] 新入口及回归测试通过：新增测试先验证失败，补齐实现后新旧筛选、模型资产和控制器共173项通过。
- [ ] 不可变部署、nohup启动及GPU实测推进确认。
- [ ] 最终筛选完成及完整归档。
