# Qwen3-4B Block3-First Validation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 按用户2026-09-19的新顺序验证4B学生，增加Step150完整保存和评测。

**Architecture:** 新建独立、固定范围的ml2队列，复用现有Revisiting训练器、
历史Block3 loss、评测/评分/验收函数。旧部署和实验保持不变。

**Tech Stack:** Python、Bash、VERL/FSDP、vLLM、ModelScope、pytest。

## 已批准的实验合同

- 学生为官方ModelScope `Qwen/Qwen3-4B-Base`，revision
  `bbd6fc8d23e8788d987b7b970cbb7bd31c826e38`，下载原始BF16权重，不用量化版本。
- 教师沿用ml2 `models/Qwen3-4B-Base-GRPO`，依据历史1.7B Token实验清单
  核验本地哈希；这是公共研究者checkpoint，不能写成Qwen官方后训练模型。
- 顺序：Block3恢复门禁 -> Block3正式200步 -> 完整评测200/150/100/50 ->
  零步学生完整评测 -> Token恢复门禁 -> Token正式200步 -> 完整评测200/150/100/50。
  门禁不是正式实验或初始benchmark评测。初始模型始终为同一份只读原始权重。
- 两正式组均从原始4B-Base独立初始化，不使用探针、Block3或Token训练后的权重。
- 保留历史1.7B训练提示（要求think标签、template不传enable_thinking）和
  评测提示（不要求标签，显式false）的差异。该例外仅适用于此新实验。
- 复用现有DAPO原始1,791,700行pool，不重抽文件/去重；train SHA
  `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`。
- seed21、200步、4 prompt x 8 nominal rollout、mini32、PPO epoch1、LR2e-6、
  prompt2048/response16384、temperature1、top-p0.9、micro-batch1/4/1、vLLM0.6。
- 历史每请求固定seed可能使同题8条训练输出重复。这是共同历史条件，不在本轮
  静默修改，不把6400条称为6400个独立样本；评测仍使用独立的seeds21..28。
- Token为k1；Block3为固定非重叠k3、mean advantage、joint PPO ratio、原block
  reduction。不加入滑窗、归一化修正或其他算法变化；先准备并比较两组配置。
- 保存50/100/150/200全部四rank模型/optimizer/scheduler/RNG及dataloader状态，
  不删除checkpoint。每组正式训练结束后逐个合并评测，避免中途评测改变训练状态。
- 每个模型完整评测MATH500500题、AIME24 30题、AIME25 30题、AMC23 83题，
  每题8次，seeds21..28、temperature1、top-p0.9、max16384，历史外部grader
  SHA `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
  共9个模型视图、46,296条评测输出；按benchmark独立报告Avg@8/Pass@8、格式和截断。
- 每步保存实际训练rollout（含token、真实engine结束原因、实际训练mask），
  保存全部评测raw archive，保留41个诊断点及位置热图。
- 先核验Llama前驱已完整结束和四GPU空闲；仅ml2，禁止访问train。
  任何OOM/资产/验收失败停止队列，保留失败证据，不自行降长度、改batch、换seed或重试。

## Task 1: 资产身份

新增 `scripts/prepare_qwen4_assets.py` 和 `tests/test_qwen4_assets.py`。
先验证错误hash/source/缺文件被拒绝，再下载固定ModelScope snapshot并核验
tokenizer共有映射、历史教师额外token例外、32k配置。只读保护原始学生和教师。

## Task 2: 历史Qwen提示与完整存档

新增 `qwen3_historical17_v1` 路径，保持原Math模板、原chat渲染、原stop配置和
原EOS mask。现有retain metadata同时改变mask，不能直接用它声称仅增加日志。
分离该新路径的mask选择，分别保存真实generation长度与完整训练张量mask。
CPU测试验证旧/新实际prompt IDs和mask一致；探针在GPU验证实际路径。
修改限于 `opd_ext/math_protocol.py`、launcher、rollout采集与相应patch/manifest。
不得修改core_algos、advantage或诊断定义。

## Task 3: 固定顺序队列

新增 `scripts/run_qwen4_pair.py` 和 `tests/test_qwen4_pair.py`。
先测试顺序、50/100/150/200覆盖、两组初始权重一致、失败不推进，随后实现。
复用现有launcher、checkpoint/resume audit、eval raw审计、绘图及prompt-order比较；
不借修改旧队列全局常量来实现新实验。初始评测即使后执行也不改原始模型。
两组共享配置预先冻结；报告逐benchmark和逐checkpoint，不挑最佳step替代200。

## Task 4: 验证与部署

运行新测试及历史protocol/loss/archive/queue回归、脚本语法/编译检查。
复核patch可以从固定上游重建，更新runtime文件哈希。独立代码审查后提交并同步GitHub。
使用现有不可变部署脚本，只创建新release，不替换旧runtime或current链接。
服务器nohup启动，记录commit/PID/log；确认新Block3探针/正式训练进度，准确区分阶段。
完整checkpoint恢复和实际rollout审计通过后才能启动正式训练。
