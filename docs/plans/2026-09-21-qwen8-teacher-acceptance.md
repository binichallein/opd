# Qwen8 Teacher Acceptance and Qwen4 Paired Training Implementation Plan

> Use the existing immutable training implementation; new controls must not replace old deployments.

**Goal:** 验收官方 Qwen3-8B-Base 的教师能力，通过后训练原始 Qwen3-4B-Base 的 Token OPD / Block3 Mean 对照。

**Architecture:** 独立 ModelScope 资产、固定题目能力门禁、真实 GPU 更新/恢复门禁、串行对照训练。任何资产/覆盖/数值检查失败或能力证据不足都停止，不自动更改标准或重试正式任务。

**Tech Stack:** Existing Python controllers, vLLM 0.11, frozen Revisiting OPD runtime 0f9161f02f08287fb07f0375ad0a6bda81133ff0, historical grader, ml2 only.

## Authorization and Invariants

- 用户2026-09-21选定8B-Base，要求先验收教师再启动训练。无权使用私人模型或访问train。
- 新学生仍为官方原始4B-Base，不能从已训练Token/Block3权重续训。
- 当前公开4B-GRPO教师仅作能力参考；8B必须优于学生，不要求超过4B-GRPO。
- 使用当前completion boxed提示，显式thinking=False，不使用chat template；EOS151643。
- 新比较改变教师身份；相对旧4B-GRPO实验同时改变规模和后训练，不能称纯规模消融。

## Frozen Capability Gate (Before Inspecting Any New Model Output)

1. 从同一DAPO池按题目去重，剔除四个benchmark的空白归一化重合题、同题不同答案的全部歧义组，以及历史seed21/200步正式4B实验实际访问的题目。按带版本盐的题目SHA256排序取64题。不修改训练文件或训练采样顺序。新模型生成前的数据预检发现首17917行存在12个空白归一化后答案冲突组，排除清单必须保存，不能任意选择其中一个答案。
2. 这是本次训练计划之外的诊断题，不保证对教师预训练/历史RL未见过，不是新的benchmark或完整eval。保存题目、答案、来源行号、排除集合和数据hash。
3. 原始4B学生、8B-Base候选、旧4B-GRPO参考分别生成每题2条。temperature1.0/top_p0.9/top_k=-1/max_tokens16384，同题同sample独立确定性seed，输入token一致，原始token/采样参数/真实finish reason/logprob全部保存。
4. 取固定前32题的学生sample0，将回答截到中点，且在第一个boxed答案前截止。三模型在完全相同的学生前缀上各继续生成2条；总response预算仍为16384，不能给教师额外token。保留前缀和新续写，拼接后由历史grader计分。
5. 能力通过标准预先固定：8B独立解题平均正确率比4B学生至少高5个百分点，题目配对bootstrap95%下界大于0；前缀续写正确率差非负。64题证据不足则记inconclusive，不能称证明教师无用，也不自动扩样本至通过。
6. 健康标准：8B独立/续写各自length-stop比例不超过10%，且不比对应学生高超过5个百分点；周期重复不超过5%；未提取到有效boxed答案不超过25%，且不比学生高超过5个百分点。记录think标签但不把Base flag当作生成保证。覆盖缺失/非有限logprob/错误输入/错误grader一律拒绝。
7. 4B-GRPO参考结果单独报告，不与不同模型结果混合成一个分数。所有比较同时给样本数和原始计数。

## Engineering Gate and Formal Training

- 仅能力门禁通过才启动训练式探针：Token和Block3各从原始学生更新到Step1并保存完整状态，再退出恢复至Step2。不得用探针权重初始化正式组。
- 使用旧冻结训练代码，通过新的MATH_TEACHER路径指定8B；新manifest/run_card保护所有公共设置。不同大小教师的模型配置、tokenizer、分片和诊断实际运行检查必须通过。
- 两组200步，先Token再Block3，seed21/LR2e-6，4题x8条/步，actor/logprob/ref micro1/4/1，vLLM0.6，16K response，历史PPO ratio与损失归一化原封不动。
- 保留Step50/100/150/200所有完整模型、四rank优化器/调度器/RNG/采样器状态；不删除任何历史产物。每步32条真实rollout及原有熵/重合率/重合质量/advantage/梯度/符号翻转/leakage/截断位置热图数据全部保留。
- 训练后验收并画图；本次控制器不自动插入新的benchmark评测。完整后续评测沿用四任务历史grader和Avg@8/Pass@8独立计分。
- 服务端nohup，独立不可变控制release，独立可执行tmpfs缓存。异常保留日志并停队列，不全局杀Ray，不覆盖旧实验。

## Implementation Tasks

1. 新增 `scripts/prepare_qwen8_teacher_assets.py` 和对应测试：ModelScope固定revision、文件完整性、可恢复下载、tokenizer一致性。
2. 新增 `scripts/qualify_qwen8_teacher.py` 和对应测试：冻结抽题/请求/前缀/评分/门禁，GPU worker保存全部生成。
3. 新增 `scripts/run_qwen8_teacher_pair.py` 和对应测试：不可变部署、资格门禁、两组恢复探针、正式训练、指标和热图验收。
4. CPU测试、独立代码复核后提交和部署。先启动资产下载与能力验收，确认进程脱离SSH、首批rollout和覆盖进度；正式训练只由成功门禁触发。

## Evidence Status

- [x] 选择官方8B-Base；ml2四张A10080GB空闲已核验。
- [ ] 固定资产下载与校验。
- [ ] 教师独立解题/学生前缀续写验收。
- [ ] 两种方法更新与恢复验收。
- [ ] Token正式训练与产物验收。
- [ ] Block3正式训练与产物验收。
