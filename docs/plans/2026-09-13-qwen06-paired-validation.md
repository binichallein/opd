# Qwen3-0.6B 配对验证执行计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 ml2 上依次执行 Qwen3-0.6B 的 Token OPD、原版 Block3 mean，
用一致的训练和完整评测合同检验 Block3 的跨学生容量泛化。

**Architecture:** 复用现有 Revisiting OPD loss、训练启动器、诊断、评测和
历史 grader。新增范围固定的服务器队列，只负责资产身份、前驱等待、
配对门禁及 48 小时预算，不修改 loss 或覆盖任何旧 runtime/run。

**Tech Stack:** Python 3.12、Bash、PyTorch/VERL/FSDP、vLLM、pytest。

## 已批准的科学合同

用户于 2026-09-13 批准：仅先做 0.6B，Token OPD 在先，Block3 mean 在后，
尤其注意实验条件对齐。暂不做 4B 学生、不加 seed、不开发新权重方法。

- Student：官方 `Qwen/Qwen3-0.6B-Base`，固定 Hub revision
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`。
  safetensors SHA-256：
  `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`。
- Teacher：现有 `models/Qwen3-4B-Base-GRPO`。它是研究者公共 checkpoint，
  不是用户私有模型或 Qwen 官方后训练发布。继续锁定本地权重哈希；旧 Hub
  revision 未记录，不能补造或自动换成网站最新权重。
- 复用 `data/math_opd_dapo17k_hf_full_eval4`，raw pool 1,791,700 行，
  train SHA `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`。
  不更改、重抽取、去重数据文件，不把行数称为唯一题目数。
- 两个正式 run 均从同一 Base 独立初始化，seed21，200 optimizer steps；
  每步 4 prompt x 8 nominal rollout；mini-batch32、PPO epochs1、LR2e-6，
  prompt/response 上限2048/16384，temperature1、top-p0.9；micro-batch1/4/1、
  vLLM0.6、四张 A10080GB；其余继承现有固定 Block3 训练设置。
- 唯一科学处理差异为 `token_opd` 与 `block3_mean`：前者 k1，后者固定
  offset0、k3、mean advantage 和 joint ratio。不得偷换成新的归一化或
  Sliding3 实现。训练前逐字段比较共享 run card、模型/数据/代码哈希；
  训练后比较41个诊断点的实际 prompt batch 哈希。回答随策略分叉是预期现象。
- 两方法分别通过 Step1 保存、退出、恢复 Step2 的完整形状门禁，随后
  正式训练从 Base 重新开始。核对四 rank 模型/optimizer/scheduler/RNG、
  dataloader 状态和恢复日志；不声称 vLLM 内部采样流可逐位恢复。
- 每组 Step50/100/200 保存全部训练状态并完整评测：MATH500500、AIME24 30、
  AIME25 30、AMC23 83；n8、seeds21..28、temperature1、top-p0.9、max16384，
  thinking关闭。每轮5144条、共30864条；不减少任务或采样数。
- 主 grader 为历史 SHA
  `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`；
  保留内置评分和原始回答，物理行 JSONL 修复必须包含在新 runtime 中。
- 主终点 Step200 的 Block3-Token macro Avg@8/Pass@8。任务内整题配对
  bootstrap10000次，seed20260913。一个 training seed，CI不覆盖训练方差。
- 原有熵/overlap/信用分配/grad norm/截断指标和位置热图不取消。
  所有 checkpoint、失败记录保留，不自动清理，不因低分换 seed 或改参。

### 已核验的 tokenizer 限制

0.6B 与历史 1.7B 学生的 BPE、added token、prompt 模板一致。
教师 tokenizer 序列化采用不同版本的 merges 格式，使用 tokenizers 库
规范化后，BPE及共有 token 映射一致；但教师额外注册 ID151665..151668
对应 `<tool_response>`、`</tool_response>`、`<think>`、`</think>`。
这是原教师已有的差异，不是新0.6B引入。新实验不改教师、词表、mask 或
原有评分实现，以保持对照；只能称“共有 token 对齐”，不能声称 tokenizer
完全相同。此限制需随新旧实验共同披露，不能把新结果单独归因于完全无
tokenizer mismatch 的蒸馏环境。

## 顺序、预算与隔离

前驱仍是 `20260911v2r1_sliding_window_seed21_ml2`，必须完整成功并释放
队列锁后才允许新 GPU 工作。空闲一张或三张卡不代表前驱已经结束。
前驱失败时停止等待并报告，不替它自动重试；绝不连接 train。

新目录：`$ROOT/runs/20260913v1_qwen06_pair_seed21_ml2`；恢复门禁放在
`probes/` 子目录，正式 run 为 `token_opd/`、`block3_mean/`。
执行顺序为 Token gate -> Token 正式训练/三轮完整评测 -> Block3 gate ->
Block3 正式训练/三轮完整评测 -> 配对比较/报告。两组正式配置提前一致性审计。

用户给新增实验48个四卡机时，即192 GPU小时。计时从前驱完成后新阶段
开始，包含门禁、训练、eval、合并和阶段内开销；准备公共资产不占用 GPU，
等待前驱不计新增预算。硬时限应预留进程组清理时间；超时停止自己的作业，
保留最近 checkpoint，标记未完成，不删数据、不缩 eval、不自动越过预算。
不能保证48小时必定完成；更不能将残缺结果当作配对结论。

## Task 1: 固定资产及对照合同

Files: `scripts/prepare_qwen06_assets.py`, `tests/test_qwen06_assets.py`。
先写 SHA/版本/词表不匹配及拒绝覆盖测试，再实现固定官方模型下载和
源清单验证。只下载非代码文件，核对公开模型 SHA，并记录教师本地哈希。
对照两组使用同一路径、revision、tokenizer和prompt模板。

## Task 2: 服务器队列和预算

Files: `scripts/run_qwen06_pair.py`, `tests/test_qwen06_pair.py`；有限修改
`scripts/launch_revisiting_block_opd_formal_train.sh` 与 `scripts/run_window_queue.py`。
先写测试覆盖方法顺序、全部共享字段、前驱失败/未完成、禁止4B/新seed、
完整eval、deadline、自己的进程组清理、拒绝覆盖失败作业。
为启动器增加严格限定ml2的本地准备传输；复用其command/run-card生成，
不重新实现训练命令。新队列使用已提交的独立 runtime，不改旧运行目录。

## Task 3: 恢复门禁与报告

复用checkpoint/eval审计、历史重评分、paired comparison及现有HTML生成器。
在queue中逐步检查退出码；验证保存的scheduler/RNG/data状态以及真正的
resume日志。任一失败均关闭队列，禁止隐式自动重试。

## Task 4: 验证、部署、启动

运行 `/home/tyf/miniconda3/envs/vllm/bin/python -m pytest tests -q`、Bash语法检查、
Python编译检查与独立代码复审。提交后部署逐文件校验的不可变版本，使用
nohup启动服务器队列，记录PID/日志/commit/等待状态；核实不会抢占前驱。
预检/运行证据另记在 `docs/results/`；计划本身不代表训练已开始或已通过。
