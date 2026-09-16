# 0.6B Token 完整评测后启动 Block3 mean

用户于2026-09-17明确授权：评测当前这对师生的Token OPD效果，然后开启
Block3 mean训练。本授权取代此前的评测/Block3暂停；不恢复旧thinking队列。

## 固定范围

- 只访问ml2。Token来源是已完成的
  `20260913v4_qwen06_nonthinking_token_seed21_ml2/token_opd`。
- 新顺序队列：`20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2`。
- 评测Token Step200、100、50，然后评测官方0.6B Base和原4B Base-GRPO教师。
  不根据某个checkpoint的分数跳过其他checkpoint，不做小样本替代。
- 四套数据分别完整评测：MATH500=500、AIME24=30、AIME25=30、AMC23=83；
  每题8条，seeds21-28，temperature1、top_p0.9、max_tokens16384，thinking
  显式关闭。每个模型共5144条，5个模型视图共25720条。
- 直接调用历史grader，SHA256
  `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
  原始生成、逐题判分、每题八条覆盖、题目/答案/prompt一致性和汇总分数都检查。
- 各benchmark分别报告Avg@8、Pass@8及Token相对未训练学生的百分点差异；
  不给跨benchmark合并分。教师作为参照，不声称未提供其他信息的性能上界。

## Block3对照

- 使用Token训练时的同一冻结runtime `0ce73aa42d7f734b9d34c379f474458bb6d48771`，
  不更改loss实现、tokenizer、模型、数据或采样行为。
- 唯一方法变化为k1/sum改为k3/mean；固定非重叠block，不是random/sliding。
- 0.6B官方Base重新初始化，不能从Token或恢复探针权重开始。
- 原DAPO池1791700行、seed21、200步、4 prompts/step、每题名义8条、
  lr2e-6、micro-batch1、vLLM0.6、prompt2048/response16384均与Token一致。
  延续原采样seed行为，不把名义8条称作8个独立样本。
- `math_eval_nonthinking_v1`、实际停止token集合、response mask、每步完整
  rollout及Step1/每5步观测完全保留。checkpoint50/100/200全部保留，禁止剪枝。
- 启动前严格比较两组run_card和输入文件哈希；仅允许方法、实验名和输出路径
  等已列明差异。先做Block3 Step1保存/Step2恢复门，再启动正式200步。
- 所有Token/参照评测成功且完整性审计通过后才进入Block3 GPU训练；模型
  分数高低不是跳过已授权对照实验的理由。本队列不自动启动Block3完整评测。

## 数据安全和实施

1. [x] 新增独立控制器和失败测试，沿用现有评测/训练/恢复检查组件。
2. [x] 校验旧runtime、模型/数据/grader与Token checkpoint；所有新产物写新目录。
3. [x] 完成正式Block3配置生成和配对差异审计，尚不使用GPU训练。
4. [x] nohup启动顺序队列，确认Token Step200完整评测实际进入GPU生成。
5. [ ] 全部评测完成后写独立计分报告，自动进行Block3恢复门和正式训练。

发生非零退出、结果不完整、哈希变化或配对不一致时停止队列，不自动覆盖/
重跑，不全局kill其他GPU任务。源Token checkpoint只读；HF权重合并到新目录。
编排代码使用独立冻结版本，训练仍固定使用上述旧runtime，分别记录来源。

启动实测与后续检查入口见
[启动记录](../results/2026-09-17-qwen06-eval-block3-startup.md)。
