# Token OPD 历史截断率对比与轨迹保存要求

## 结论与范围

2026-09-13 在 ml2 重新读取两次正式 Token OPD 的逐步训练日志。
之前 Qwen3-1.7B-Base 的全程平均截断率为 13.625%，本次
Qwen3-0.6B-Base 为 39.125%，增加 25.5 个百分点，约为之前的 2.87 倍。
这里的“之前”专指 ml2 上 2026-07-12 的正式 baseline；没有访问 train，
也不把 probe、Block3 或评测阶段的数据混入统计。

| 指标 | 之前：1.7B Token OPD | 本次：0.6B Token OPD |
|---|---:|---:|
| Step1-50 平均截断率 | 14.0% | 33.5% |
| Step51-100 平均截断率 | 13.0% | 42.0% |
| Step101-150 平均截断率 | 12.0% | 40.5% |
| Step151-200 平均截断率 | 15.5% | 40.5% |
| 全部 200 步平均截断率 | 13.625% | 39.125% |
| 最近 20 步平均截断率 | 16.25% | 45.0% |
| 平均回答长度/token | 4119.75 | 7498.02625 |
| 整个 batch 都触顶的步数 | 0 | 9 |
| 第一次更新前 rollout 的截断率 | 25% | 50% |
| 触顶回答占全部生成 token 的比例 | 54.19% | 85.49% |

之前也存在截断，但没有本次严重。两次从第一步就有差异，支持进一步排查
学生初始策略与当前提示/停止协议的适配，而不是把差异全部归因于后期更新。
这不是“参数量小必然导致截断”的因果证明，也不是方法效果的 benchmark 结论。

## 统计与可比性

- 使用完整 200 步 `response_length/clip_ratio` 的算术平均；两份日志各有
  200 个唯一训练 step，没有把 `perf/time_per_step` 误当作训练步数。
- 指标定义是有效响应长度等于响应张量宽度的样本比例。本次核验中，两组
  所有正截断率对应的 `response_length/max` 均为 16384；按长度触顶率解释，
  不冒充生成引擎的原始 `finish_reason`。
- 同为 4 prompts/step、名义每题 8 个 rollout、200 步、seed21、最大响应
  16384。教师、数据路径/训练文件哈希、学习率、micro-batch、vLLM 占比等
  共同字段在 run card 中一致。组内 rollout 不应未经检查就视为独立样本。
- 学生不同，代码版本也不同：旧版本为
  `9b3b8b76bcdf02d0a4cfe4720cecb24bc7203977`，本次冻结版本为
  `ec0a7a950540d7f4753c08b18bc26c544f6a7cde`。这是一项历史描述性对比，
  不是排除了所有混杂因素的模型规模消融。
- 触顶回答的 token 占比按
  `sum(clip_ratio * 16384) / sum(response_length_mean)` 计算，利用每步批量
  相同的条件。这是生成 token 的比例，不是有效梯度贡献比例。

## 后续代码审计：版本不同不等于 Token loss 改变

用户追问代码差异后，直接比较 ml2 上上述两个冻结 deployment，并重新
验证两次运行保存的 `script_hashes.sha256`。两组全部通过；各自 runtime
manifest 中的 1248 个文件也全部匹配部署内容。

上游运行目录只有四个文件发生变化：

| 文件 | 改动 | 本次 fixed Token 是否启用新算法 |
|---|---|---|
| `verl/trainer/config/ppo_trainer.yaml` | 增加 window mode/seed 配置 | 否，mode=fixed |
| `verl/trainer/ppo/core_algos.py` | 增加 random/sliding loss 分支 | 否，仅非 fixed 时调用 |
| `verl/workers/actor/dp_actor.py` | 传递窗口参数，增加窗口模式检查 | 否，fixed 的窗口参数为空 |
| `verl/trainer/ppo/ray_trainer_multitask.py` | 窗口状态恢复、offset 调度、诊断分派和元数据 | fixed 不创建随机 offset 调度；沿用原诊断计算 |

`opd_ext/window_supervision.py` 是新增扩展；固定 Token 不进入其窗口目标。
原 `opd_ext/diagnostics.py` 没变。数据集加载、rollout loop、vLLM rollout、
FSDP worker、长度掩码与截断指标文件均未变化。
`core_algos.py` 中原 advantage、固定 block 聚合和 loss 归一化逻辑未变化。
block size=1 时，聚合函数直接返回原 token 张量。

逐项解析两份训练日志中最终展开的配置后，除学生及其关联 tokenizer 路径、
运行名称/输出路径外，只增加了 `opd_window_mode=fixed` 和
`opd_window_seed=910021`。后者在 fixed 模式不使用。
命令行显式加入 `ppo_epochs=1`，但旧配置展开后同样为 1，不是实际改变。
两次记录的 Python 3.12.13、PyTorch 2.8.0+cu128、Transformers 4.57.6、
vLLM 0.11.0 一致；这不等于声称所有未记录依赖或硬件状态均已验证相同。

只读 CPU 验算直接从两个部署的 AST 提取原 loss/helper 函数，使用固定
独立 CPU generator seed=20260913，覆盖 float32/float64、长度 1/2/7/17、
完整/尾部 padding/内部空洞 mask、log-ratio 扰动 0/0.1/1.5：

- Token k=1、sum：72 组输入全部通过。
- 固定 Block3、mean：另外 72 组输入全部通过。
- 四项 loss 返回值和对 sampled log-prob 的梯度逐元素完全一致，最大差异 0。
- 仅验证合成输入的策略 loss，不是完整模型梯度或 GPU 训练的逐位复现证明。

因此，应把此前“代码版本不同”的保留意见细化为：版本确实不同，但本次
审计未发现改变 fixed Token 训练目标、采样或截断统计的有效路径变更。
不能仅凭版本号不同解释 13.625% 到 39.125% 的差异，也不能由此直接证明
差异完全来自学生参数量。

## 原始证据

共同根目录：
`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs`。

旧日志：
`20260712v1_token_opd_replication_seed21_ml2/token_opd/logs/nohup.log`

- 长度：579356 bytes。
- SHA256：`221bd2fa13dfbc0ea8c0be57ef87d0cbff69493315362c28287317da7dce9bf0`。

本次日志：
`20260913v1_qwen06_pair_seed21_ml2/token_opd/logs/nohup.log`

- 长度：596058 bytes。
- SHA256：`9552162ff80de0b39f08e7491d6da458bccbf0d2a0d9021a2ec416b51ac5b31a`。

配置证据：上述两个 `token_opd/run_card.json`。原始日志和模型未修改，
没有启动额外 GPU 实验。

## 后续必须保存训练轨迹

用户明确要求：以后训练必须保存 rollout 轨迹。该要求已写入 `AGENTS.md`。
保存范围包括每步、每条实际训练响应的 prompt、原始 token IDs、保留特殊
token 的文本、长度/掩码、样本与 step 标识，以及可获得的真实停止原因。
采样配置和 seed 通过 run manifest 关联。不能只保存汇总指标，不能把
checkpoint 重新采样的回答标为原始训练轨迹，也不能在恢复时覆盖旧轨迹。

本次已结束的 0.6B Token 原始训练轨迹无法事后补回。现有冻结脚本默认
`trainer.rollout_data_dir=None`；已有文本导出函数还使用
`skip_special_tokens=True`，仅打开这个开关不足以满足特殊 token 排查需求。
本记录只确立后续新训练的启动要求，没有热修改当前队列或宣称其已经合规。
对已排队任务补充记录时，必须先形成可追溯的仅日志变更并验证，再启动。

## 用户要求关闭 Thinking 后的协议核验

用户进一步明确要求：之后所有训练不要开启 think 模式。不能将此前
评测中的 `enable_thinking=false` 当作训练已经关闭的证据。

2026-09-13 再次只读核验 ml2 的冻结代码和最终展开训练配置：

| 实验 | 训练模板 kwargs | 数学环境指令 | Step50 评测 thinking |
|---|---|---|---|
| 1.7B Token，20260712v1 | `{}`，未显式关闭 | 要求在 think 标签内推理 | false |
| 1.7B Block3，20260711v2 | `{}`，未显式关闭 | 同上，冻结版本相同 | false |
| 0.6B Token，20260913v1 | `{}`，未显式关闭 | 同上 | 评测代码显式 false |

前两组冻结版本为 `9b3b8b76bcdf02d0a4cfe4720cecb24bc7203977`，第三组为
`ec0a7a950540d7f4753c08b18bc26c544f6a7cde`。两个版本的
`agent_system/environments/prompts/math.py` 都包含：

```text
You must conduct reasoning inside <think> and </think> and give the final answer within \boxed{}.
```

`agent_system/multi_turn_rollout/rollout_loop.py` 将配置中的
`data.apply_chat_template_kwargs` 直接传给 tokenizer。各正式训练日志
记录该字段为 `{}`，不是 `{"enable_thinking": false}`。

已检查两个 Base 模型本地 tokenizer 的实际模板：当且仅当显式传入
`enable_thinking=False` 时，在 assistant 前缀附加一个已经闭合的空块：

```text
<|im_start|>assistant
<think>

</think>

```

这个空块表达“跳过 thinking”，不是要求模型开始输出思考；不能仅凭
prompt 中出现 think 字样判断模式是否打开。Base 模型是否遵守该控制
也要通过实际输出验证，不能由参数名保证。

早期 `scripts/clean_opd_train.py` 的 `--enable-thinking` 默认 false，
但当前 Revisiting OPD 环境不是这条入口。未逐一核验的更早实验不作
“全部都关了”或“全部都开了”的断言；本次没有访问 train。

后续协议同时要求显式 false 与删除环境中的 think 指令，并检查实际
渲染后的 prompt/token IDs。旧版 `no_think` 诊断仅删除后者，没有传入
false，因此其结果不能标为“完整 non-thinking 模式”的验证。

此处是用户新约束与历史审计记录，没有热修改冻结训练脚本或恢复队列。
后续新 Token/Block3 对照必须使用相同的新协议，保留旧结果且单独命名，
不能只给 Block3 关闭 thinking 再与旧 Token 直接归因比较。
