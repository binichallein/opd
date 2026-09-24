# 旧版1.7B / 4B-GRPO的B、C组件消融

## 授权与边界

用户明确选择旧配置，因为历史Block3收益最明显。本轮只新增B、C，不重训A/D、不切换新prompt。
仅使用ml2，四张A100；保留全部checkpoint与轨迹，不删除、不自动重试、不根据分数改参数。
旧A/D仅作为历史对照，不宣称是同一新运行环境下重训的四臂析因实验。

## 数学定义

令a_t为停止梯度的teacher-old_student log-prob差，n_j为第j块有效token数，
bar_a_j=sum(a_t)/n_j，r_t=exp(current_t-old_t)，R_j=prod(r_t)。
C(r,a)沿用历史PPO/dual-clip surrogate；T为当前micro-batch的有效token总数。

- B (`adv3`, `adv_only`): L_B=-sum_j sum_{t in B_j} C(r_t,bar_a_j)/T。
- C (`joint3`, `joint_tokenmean`): L_C=-sum_j C(R_j,bar_a_j)/T。
- 历史D: L_D=-sum_j n_j C(R_j,bar_a_j)/T。
- 历史A: 原始逐token优势及ratio。

C按每个块的实际n_j消除D的额外倍率，不是无条件除3，也不改变学习率。
代码保留原mask和loss数值稳定epsilon。B/C目标仅支持fixed、mean、token-mean、sampled PPO。
旧策略处未裁剪时B/C对logprob梯度相同；完整块D梯度为B/C的3倍。
这不等价于Adam参数更新3倍。当前PPO仅1epoch，B/C的实际差异可能很小；这也是有效结果。

## 冻结协议

- 学生：历史manifest中的官方Qwen3-1.7B-Base原始权重，不是旧实验训练后权重。
- 教师：公开lllyx/Qwen3-4B-Base-GRPO，与历史artifact manifest逐文件校验。
- DAPO文件SHA256 `cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf`。
- seed21，200步，每步4题x8回答；lr2e-6，PPO mini-batch32，epoch1，梯度裁剪1。
- prompt2048，response16384，temperature1，top_p0.9；actor/ref micro-batch1，logprob4；vLLM显存0.6。
- 历史训练：ChatML + 必须在think标签内推理；模板kwargs不传enable_thinking；EOS151643，额外stop=[]。
- 历史请求seed：每个请求重复seed21，明确保留该历史行为，不推荐用作新实验默认。
- 历史EOS-derived mask不改成长度mask；保存完整training mask和生成长度，便于后续诊断。
- 历史评测：原聊天题面、enable_thinking=False、stop=[151645,151643]，固定历史grader。
- 主动保留历史训推差异，不声称已统一；不同日期依赖环境可能不支持bitwise重放。
- 诊断与旧版同为interval5，含Step1；新增完整rollout存档，以及额外Step150保存。
- 沿用已获授权的完整状态保存策略：50/100/150/200全保留，模型、四rank优化器/RNG、scheduler、dataloader等。

## 队列与验收

每组CPU Ray预热 -> Step1保存退出 -> Step2完整状态恢复验收 -> 从原始学生重新初始化正式200步
-> 全状态/6400条训练轨迹审计与热力图 -> 完整评测200、150、100、50。
B全部完成后自动进入C。每次评测643题x8=5144输出；总计8个新增完整评测。
MATH500、AIME24、AIME25、AMC23分别报告Avg@8、Pass@8、格式、重复/截断，不合并macro。
每步数据顺序与已留存的同seed DAPO轨迹来源交叉核验；B/C最终对齐全部6400条输入与采样条件。
历史A/D仅有50/100/200，150只作B/C内部比较。

## 实施进度

- [x] 历史run card与原始模型路径、seed、micro-batch等核对。
- [x] 新loss测试先失败，实现后数学/梯度及原loss回归17项通过。
- [x] 新队列测试先失败，新增及历史控制器回归88项通过。
- [x] 不可变部署、ml2实际环境97项CPU测试及配置验收；另24项legacy数值回归通过。
- [x] B探针首步GPU更新、checkpoint/四rank优化器/32条轨迹验收；22:29启动恢复探针。
- [x] B保存恢复探针、64条轨迹及四rank完整状态验收；22:43自动启动正式B。
- [x] B正式训练首步核验：22:55 Step1/200，32条输入/生成/mask与探针首步一致，数值有限。
- [ ] B/C全部训练、评测、轨迹及结果归档。
