# 0.6B Non-Thinking Token OPD 启动记录

## 运行身份

- 当前 ml2 尝试：`20260913v4_qwen06_nonthinking_token_seed21_ml2`。
- 冻结训练 runtime：`0ce73aa42d7f734b9d34c379f474458bb6d48771`。
- 控制器 PID3844168，2026-09-13 17:35:49 北京时间开始运行。
- 根目录：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
- 状态文件：上述根目录的 `runs/<运行身份>/queue_state.json`；先读实时状态，
  不根据文档中的历史阶段重复启动。
- 旧 v1 完整评测队列仍保持停止；没有启动 Block3，没有删除旧模型或日志。

## 配置与当前验收

完整方案见 [训练方案](../plans/2026-09-13-qwen06-nonthinking-token.md)。
学生为官方 Qwen3-0.6B-Base，教师为现有公共 Qwen3-4B-Base-GRPO；原
DAPO 数据、seed21、200步、micro-batch1、vLLM占比0.6和Step50/100/200
checkpoint保留策略不变。没有增加SFT阶段。

- 本地全库525 passed、2 skipped；源代码编译和Shell语法检查通过。
- 上游patch的反向应用检查通过；新patch中带入的上游空白上下文会被
  `git diff --check`识别为空白行，非patch源码检查通过，没有因此改写上游
  无关空白。实际runtime文件全部由SHA256 manifest校验。
- 远端不支持硬链接：完整轨迹使用独占的attempt/step目录，压缩JSONL
  写完后rename发布并保存SHA256；崩溃残留目录也不会被重试覆盖。
- GPU检查已验证16题经过真实MathEnvironmentManager/TrajectoryCollector
  与评测入口的prompt token IDs完全一致；显式enable_thinking=False。
- 现有评测的全部prompt生成规则已核验：MATH500 500条、AIME24 30条、
  AIME25 30条、AMC23 83条，共643条。没有改变评测数据文件。
- GPU生成检查已完成，退出码0。16/16未新生成think标签，7/16触及16384
  上限，平均长度7486.1875。此项验证输入控制和样本行为，不证明Base永远
  遵从、不证明无推理，也不证明截断或准确率已改善。
- GPU原始输出SHA256：
  `dcb96ec8d3cbe790950266f69cc8d00ffbbbf4aab8d06b79a70b358f0fa2453a`。
- 两步四卡保存/恢复测试均退出0，checkpoint/diagnostics审计通过。
  `resume_gate.json`验证Step1和2共8个rank状态，scheduler分别为1和2，
  RNG种类及dataloader读取位置齐全；这不是逐bit vLLM重放保证。
- `rollout_acceptance.json`通过：两步各32条均完整、0新think标签，第二步
  仍有16条length stop。Step2原始轨迹SHA256：
  `fec8e1587d48b32875fd3c425c6797024468c15227baf98e495222a1d64a1cef`。
- 正式进程PID3867901，于18:06:07北京时间启动；`run_card.json`确认目标
  200步、Token k1、官方Base路径、resume_mode=disable、resume_from_path为空、
  新非thinking协议和formal轨迹目录。正式首步已完成并通过轨迹审计，训练
  继续运行；没有将探针结果混写为正式训练结果。
- Step1完整归档32条，prompt重建与SHA校验通过，0条新think标签、16条length
  stop，平均response长度8360.5。裁剪前grad norm=15.507，student entropy
  约0.106、teacher entropy约0.095、top16 overlap约0.483；所有非有限值计数
  为0，Token k1的sign flip/leakage为0。位置NPZ约1.1MB，已保留。
- Step1轨迹SHA256：
  `cf6307525fd9412494b29c54edb20aae4ff59bd32e17d853529204dc1c0d07bd`。
- 此批每题8条的长度完全重复，不能将名义32条说成32个独立样本；沿用既有
  sampling seed设置，本次没有静默改变该实验条件。
- Step1退出清理阶段出现MathMultiProcessEnv析构中的DataLoader worker killed
  异常提示，完整checkpoint保存后进程仍退出0。保留原日志并验证Step2恢复，
  不把此清理提示当作已证明的GPU OOM，也不隐藏它。

## 正式首步验收

- 正式Step1为32条，prompt IDs重建、显式false、停止集合及文件SHA均通过。
  文件为`token_opd/rollouts/formal/step_000001/raw.jsonl.gz`，SHA256：
  `33989511532ae47b35ab85d1e58654158915e8a23d4b8970d7f069e305730a97`。
- 实际更新的PG loss=0.17612464，裁剪前grad norm=15.50526333；
  student entropy=0.10599683，teacher entropy=0.09545896，top16 overlap=
  0.48317796。原有诊断全部记录，非有限值计数0，Token k1信用指标为0。
- 新think标签0/32，真实length stop 16/32，平均长度8360.5，最大16384。
  这说明显式关闭和轨迹记录已工作，但长输出截断没有自动消失；不是方法
  有效性、正确率改善或长期训练稳定性的证明。
- `diagnostics/scalars.jsonl`及`step_00001.npz`存在；正式5张诊断图已生成，
  位置热图已检查。本地副本位于
  `/home/tyf/paper/outputs/qwen06-nonthinking-startup/formal/`。
- 本地完整测试再次执行：525 passed、2 skipped，12.51秒。后续每5步继续
  保存位置观测，控制器训练结束时重绘整条曲线；当前图仅含首步。
- 正式训练仍在后台运行，目标200步，并按50/100/200保存完整状态。先查
  `queue_state.json`与`token_opd/logs/nohup.log`获取最新进度，不根据本文
  首步快照判断实时状态。

## 独立代码复查边界

- 检查器会将超过2048 token的评测提示截断后比较，而当前评测入口使用
  完整提示。因此不能将门测试推广为任意长输入都完全一致。本次另行使用
  官方tokenizer核验未截断输入：GPU16题最大170 token，保存的实际IDs全部
  与完整encode相等；AIME24最大224、AIME25最大280、AMC23最大385、MATH500
  最大818，643题均不触发此边界。新增长提示评测前必须统一实际截断策略。
- 控制器的进程内audit依赖runtime导入路径。已读取PID3844168真实环境，
  `PYTHONPATH`包含冻结runtime根目录及external/revisiting_opd，不会触发
  本次启动的缺失路径风险。未来入口仍需保留该环境，不能假设仅给子进程
  设置PYTHONPATH就能修复父进程导入。
- 以上为本次运行的实测适用边界，没有在线改写冻结runtime。

## 轨迹与诊断位置

- 独立GPU检查：`gpu_gate/input_contract.json`、`gpu_gate/raw.jsonl`、
  完成后的`gpu_gate/summary.json`。
- 恢复测试：`probes/token_opd`，两个attempt分别为`probe1`和`probe2`。
- 正式训练：`token_opd`，从官方Base重新开始，attempt为`formal`。
- 每步原始轨迹：`token_opd/rollouts/formal/step_XXXXXX/raw.jsonl.gz`，
  含真实prompt/response IDs、特殊token原文、finish/stop、采样参数、mask、
  rollout logprobs、数据行信息、group/traj/step/run/source身份及文件SHA。
- 原有过程标量/位置数据：`token_opd/diagnostics`，Step1及每5步保存。
- 训练日志每步还记录`rollout_archive/count`、`length_stop_rate`和
  `generated_think_tag_rate`。不将长度触顶与真实finish reason混为一谈。
- 诊断热图：`token_opd/figures`，控制器会在训练完成后生成；启动核验期间
  也可用冻结版本的`analyze_single_opd_diagnostics.py`生成已有step图像。

## 历史失败尝试

- v2 / 8a0a2d3：独立测试输入缺少batch字段，生成和训练均未开始。
- v3 / 7724141：CPU prompt检查通过，但vLLM fork子进程无法重新初始化CUDA；
  尚未生成响应。v4显式使用spawn，仅修复测试进程启动方式。
- 两次源版本、运行目录、日志、退出码保留，未作为有效训练结果使用。
