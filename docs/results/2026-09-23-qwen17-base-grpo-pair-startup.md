# 4B-GRPO到1.7B-Base完整对照启动记录

## 授权与范围

能力测试仍保留原结论inconclusive，仅续写增益未通过，报告未改写。
用户随后明确要求“这次不管了，启动训练吧，和上次一样”。本队列据此继续训练，
不是将能力门槛伪装成通过；运行目录单独封存`training_authorization.json`。

教师为公开`lllyx/Qwen3-4B-Base-GRPO`，学生为官方`Qwen/Qwen3-1.7B-Base`。
两组从同一原始学生分别初始化，完整流程、保存策略和诊断见[冻结计划](../plans/2026-09-23-qwen17-base-grpo-blockfirst.md)。

## 已核验

- 只连接ml2；启动前四张A100空闲，历史队列均complete，不覆盖旧运行。
- 新增12项先失败再实现，相关本地CPU回归204项通过；部署后ml2核心回归81项通过。
- 不可变代码版本：`26382fcff692667269119a95b52bebceb40d3ced`。
  `deployments/<commit>`的完整文件哈希校验通过。
- **19:28:33北京时间**服务端nohup启动；PID/SID=`3144910`、PPID=`1`，不依赖本地SSH在线。
- 输入核验通过：实际collector覆盖64道筛选题和643道评测题，与教师及评测输入逐token一致。
  协议`qwen3_completion_boxed_v1`，未调用聊天模板，原生EOS151643，无额外停止token。
- 两组正式run card通过配对核验：各200步、seed21、lr2e-6，每步4题x8轨迹，
  actor/ref micro-batch=1、rollout logprob micro-batch=4、vLLM显存利用率0.6。
- 保留原历史Block3 Mean的联合ratio及损失归一化，没有更换loss或教师模型文件。
- 各组保存50/100/150/200的模型、优化器、调度器、sampler与rank RNG状态；不自动删除。
- 每步保留原始轨迹及诊断；每组结束自动生成诊断图。
- Ray CPU预热和真实worker检查通过，Ray版本2.55.1。

## 队列顺序

1. Block3保存/恢复探针，验收后正式训练200步。
2. Block3 Step200/150/100/50完整评测，然后初始学生完整评测。
3. Token保存/恢复探针，验收后独立正式训练200步，再评测Step200/150/100/50。

每次评测均为MATH500 500题、AIME24 30题、AIME25 30题、AMC23 83题，每题8次，
总计5144条；历史grader不变，各benchmark单独计分，不以macro代替。

训练文件沿用历史`data/math_opd_dapo17k_hf_full_eval4/train.parquet`，
SHA256=`cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`。
其物理行数为1791700，不能把这个数写作独立题目数。本轮没有重建文件或改变seed21源题目顺序。
每组200步消费800个题目位置，生成6400条轨迹，允许同一题重复出现。

## 启动时点状态

19:41时仍在Block3第一步探针初始化，尚无正式训练指标，不宣称保存/恢复门槛已通过。
数据任务索引已构建完成，四个GPU worker已建立通信并加载模型。
后续以`queue_state.json`、`probes/block3_mean/resume_gate.json`及正式rollout记录为准。

19:45补充：第一步探针已完成真实更新并保存checkpoint，轨迹验收通过。
32条轨迹使用32个独立请求seed，截断1/32=3.125%，生成think标签为0，平均长度1401.0625。
裁剪前grad norm=2.630036、PG loss=0.073837，student entropy=0.204719、teacher entropy=0.173542，
top16 overlap=0.683039、sign flip rate=0.137035。entropy、advantage及post-update block ratio的非有限计数均为0。
位置诊断已写入`step_00001.npz`，标量写入`scalars.jsonl`。这些是探针结果，不计为正式Step1，
也不能证明后续200步稳定或方法有效；完整恢复及正式运行状态仍需后续记录。

19:48复核：四个rank的Adam step/moments检查通过，各有29组非空参数状态和非零moment；
第一步轨迹与历史源题目/请求seed核验通过。第二步进程已于19:46:51启动，
配置明确从探针`global_step_1`恢复，`resume_gate.json`尚未生成，正式200步尚未开始。

## 证据与备份

远程完整目录：

```text
/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260923v4_qwen17_base_grpo_blockfirst_seed21_ml2
```

本地启动快照：

```text
/home/tyf/paper/outputs/qwen17_base_grpo_pair_20260923/startup_1941
```

本快照保存启动记录、固定配置、输入/配对验收及当时日志，不包含尚未完成的正式轨迹或checkpoint。
19:46追加备份了已完成的第一步探针原始轨迹及其校验文件。
19:49补充诊断和首步验收记录；本地9个受保护元数据文件及首步轨迹SHA256核对通过。
轨迹SHA256为`c668d7ca0defe70e22e20a9a691da4e02ca81dc9a7a76ab3d7e493831ca12527`。
大权重、完整训练与评测轨迹保留在ml2；最终结果需在完成后另行汇总备份。
