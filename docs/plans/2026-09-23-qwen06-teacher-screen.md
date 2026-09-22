# Qwen0.6 八组教师筛选实施方案

**Goal:** 对用户指定的八对师生进行推理能力验收，区分教师原生能力、学生输入适配和学生前缀续写能力；本轮不训练。

**Architecture:** 复用现有冻结题目、历史grader、采样及健康门槛。新增独立版本的跨协议诊断器与四卡DAG队列；旧运行代码和数据不改写。

**Tech Stack:** ml2四张A100 80GB，既有torch2.8.0/vLLM0.11.0/Transformers4.57.6，ModelScope固定资产，CPU pytest和原始输出SHA验收。

## 已确认口径

2026-09-23用户指定以下八组，顺序均为教师到学生；官方指令模型名称不带Instruct后缀。

| ID | Teacher | Student |
|---|---|---|
| b4_b06 | Qwen/Qwen3-4B-Base | Qwen/Qwen3-0.6B-Base |
| i4_i06 | Qwen/Qwen3-4B | Qwen/Qwen3-0.6B |
| g4_b06 | lllyx/Qwen3-4B-Base-GRPO | Qwen/Qwen3-0.6B-Base |
| g4_i06 | lllyx/Qwen3-4B-Base-GRPO | Qwen/Qwen3-0.6B |
| i8_i06 | Qwen/Qwen3-8B | Qwen/Qwen3-0.6B |
| b8_b06 | Qwen/Qwen3-8B-Base | Qwen/Qwen3-0.6B-Base |
| i8_b06 | Qwen/Qwen3-8B | Qwen/Qwen3-0.6B-Base |
| b8_i06 | Qwen/Qwen3-8B-Base | Qwen/Qwen3-0.6B |

用户另明确确认：主验收教师接收学生实际完整prompt和续写前缀；教师原生模板成绩另作对照。
原始0.6B学生独立生成，不使用任何OPD checkpoint。无新SFT/OPD训练，也不自动根据筛选结果开训。

## 冻结推理设计

- 复用9月21日验收已选定的64道DAPO题，selection SHA256
  `8094e5d8c97c77a654f695b13943a227bb4a1e3fe891fdb3fbce63c19de51f3b`。
  不是新盲测，不重新按模型输出挑题；既有数据排除、题目及来源哈希均复核。
- 独立解题64题x2；前32题各取该学生sample0、首个boxed前的token中途前缀，师生各续写2次。
  不重新编码前缀，不在输入中泄露标准答案；总生成预算16384，续写扣除已有前缀。
- Base学生使用统一boxed completion，无ChatML；指令学生使用原生chat，显式enable_thinking=False。
  两种学生各生成一份冻结题目/前缀，所有对应教师共享，不重复采样学生基线。
- 主比较输入token IDs严格相同。跨Base/Instruct不强求整份tokenizer文件一致，但必须检查核心词表ID一致、
  实际输入ID不越界；记录目标tokenizer未映射的输入token，不能将此类适配失败藏掉。
- 模型使用自身原生EOS；所有单元显式停止集合[151643,151645]，按引擎length/stop判断。
  这是新跨协议版本；不把旧仅151643的Base诊断当逐位等价重跑。
- 教师原生对照：Base采用completion；官方指令版及公开GRPO后训练教师使用其自带chat模板、显式false。
  不修改tokenizer/EOS配置。GRPO的原生EOS可能仍为151643，须按资产实测而非按名字推断。
- 原生教师独立解题与某个主测试的输入及采样完全相同时复用该单元，并记录引用，不重复计为独立证据。
  全部八对构成10个不同(model, student-input-protocol)组合，20个正式GPU单元，共1920条唯一诊断输出。
  另7个模型各4条256-token原生协议smoke，共28条；模型加载后核对实际输入/输出/停止符。
- sampling：temperature1、top_p0.9、top_k=-1；沿用旧sha256按question/sample/phase派生的独立请求seed。
  BF16、TP1、eager、memory0.6、max_model_len18432、max_num_seqs32。
- 保存每个原始响应token IDs、未剥离特殊token的文本、logprobs、finish/stop reason、输入身份、模型revision和采样参数。
  grader仍为历史SHA04f7；评分异常独立记为工程失败，不偷偷记为答错。

## 统计与科学边界

- 每对分别报告学生/教师独立平均正确率和续写正确率，以及teacher-minus-student的题目配对bootstrap95%区间。
  固定10000次bootstrap、seed21，不把同题两次输出当独立题目。
- 沿用门槛：独立正确率差>=5个百分点且95%下界>0，续写差>=0；教师截断<=10%、比学生高<=5个百分点，
  周期重复<=5%、缺有效boxed<=25%且比学生高<=5个百分点；非thinking条件生成think标签另列并参与健康检查。
- 原生教师正确率与学生输入条件下正确率之差单列为提示适配诊断，不与续写增益混成总分。
- passed/rejected/inconclusive是探索性筛选结果，不是蒸馏收益证明。八对共享模型、题目和学生轨迹，
  不能当八次独立重复；95%区间未做多重比较校正，不宣称整体显著。
- 后续若写成筛选创新点，需用未参与规则设计的师生/题目及实际OPD收益验证预测性，比较仅按教师绝对能力选取等基线。
  本轮不能预设创新性成立，也不使用最终benchmark分数调验收阈值。

## 实施任务

1. `scripts/prepare_qwen06_screen_assets.py`、`configs/experiments/qwen06_teacher_screen_assets.json`：
   ModelScope冻结文件元数据，新增0.6B/4B指令权重；其余资产只读复用并核对哈希。公共GRPO保留原发布者身份。
   先测试路径约束、hash损坏拒绝、已有资产不可覆盖，再下载/核验。
2. `scripts/qualify_qwen06_teachers.py`、`tests/test_qwen06_teacher_screen.py`：
   先测试精确八对、跨协议输入/续写/预算/种子、原生对照引用、完整覆盖和错误拒绝；再实现请求及封存/评分。
3. `scripts/run_qwen06_teacher_screen.py`及队列测试：单控制器、四个独占GPU槽位，依赖就绪才启动；
   独立失败不覆盖重跑、不影响无依赖其他组合；严格保留失败原始记录，最终区分科学拒绝与工程失败。
4. CPU回归、真实资产/原生模板检查、独立代码复核后提交；不可变部署，新根目录
   `runs/20260923v1_qwen06_teacher_screen_ml2`，短tmpfs缓存，服务端nohup启动并核验PPID/SID。
5. 监督首批真实GPU输出和后续并行调度；完成后本地备份、校验，GitHub同步分对汇总/图表/来源清单。

## 进度

- [x] 用户确认八对及跨协议主验收口径；现场四张GPU空闲。
- [ ] 固定资产并准备缺失模型。
- [ ] 实现与CPU/GPU协议验证。
- [ ] 启动不可变四卡队列并监督。
- [ ] 全部单元验收、报告与双端归档。
