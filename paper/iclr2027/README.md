# ICLR 2027双语论文工作目录

题目：Block-Level Credit Assignment in On-Policy Distillation。
英文为按ICLR2027匿名样式编排的论文稿；中文为对应阅读稿，不是另一次投稿。
所有实验结论须以验收完成的数据为准。不能把本目录的产生本身视为实验验证或可直接投稿的保证。

4B Base 对照的四个Token权重已全部验收，最终数据为`data/qwen4_final_20260921.json`，
表格为`generated/qwen4_publication/`。`qwen4_final/`为相同分数的早期标签版本，
publication版本仅规范模型名。旧partial缓存不用于成稿。
全文保留历史正向、负向、中止与先导实验；完整性不等于所有实验具有相同证据强度。

2026-09-23新增已完成的官方`Qwen/Qwen3-8B`教师到`Qwen/Qwen3-1.7B`学生指令版对照。
包括初始学生，以及Token/Block3各Step50/100/150/200，共9个模型视图；
四项benchmark分别报告，未混入仍在进行的4B到0.6B训练结果。
新表格及带来源哈希的紧凑汇总位于`generated/qwen17_instruct_20260923/`，
曲线位于`figures/qwen17_instruct_20260923/`。
该组使用原生聊天模板并显式关闭thinking，不与历史Base/GRPO组混成同一实验。
Step200四项Avg@8差值为正，但配对题目区间均包含零，Pass@8有升有降。

## 结构

- `main_en.tex`、`main_zh.tex`：两个入口，共用官方2027模板及同源结果。
- `content_*`：摘要、引言和全文编排；按OPSD的叙事顺序重组为引言、背景、方法、实验、相关工作、结论。
- `background_*`、`method_*`：必要背景、Block3目标和更新分析。
- `protocol_*`、`results_*`、`analysis_*`：实验设置、完整主结果和分主题分析。
- `related_*`、`conclusion_*`：相关工作定位与结论。
- `appendix_*`、`limitations_*`：实验范围、完整推导、配置、全部结果及诊断。
- `references.bib`：核验后的参考文献。
- `data/`、`generated/`、`figures/`：紧凑证据、自动表格和图；不存完整rollout或权重。
- `internal/`：内部来源映射、证据审计、合规记录。**不得放入匿名投稿包。**
- `build/`：编译中间件、PDF、文本与验收结果，不入Git。

2026-09-26全文重组保留已有图文件、生成表和分数，历史端点表移至主文；
未加入仍在进行的组件消融或后续100步实验。主文保留正负结果和关键统计边界，
逐字prompt、运行审计、详细限制等放在对应附录。官方模板已重新下载并核对一致，
核验来源和hash见`internal/compliance.md`。

同日补充教师来源：正文首次介绍4B GRPO时引用Rethinking OPD（2604.13016），
并链接现有`Thinking-Space/Qwen3-4B-Base-GRPO`模型卡；附录保留原`lllyx`仓库ID和下载revision。
主文历史表改为明确标注的两组正向Avg@8案例，0.6B与Llama结果不再占据该主表；
完整四组终点表、所有checkpoint、负结果和协议说明仍在附录E，不以结果好坏删除记录。
`history_highlights.tex`与完整`history_endpoints.tex`由同一生成器、同一数据生成。

块长选择已在方法与正文4.3.2明确交代：1.7B-Base/4B-GRPO的200步扫描比较k=1/3/5/10，
Block3的四项Avg@8高于Block5，相对Token三项提高、一项下降；因此保留短块默认值，
不称为普遍最优。正文表4直接给出四个benchmark的Avg@8 / Pass@8，
表题在上，表下注明硬件、评测seed和Block5重启及来源边界；完整协议仍见附录E.6，
未与后续同机复跑合并。`history_block_sizes.tex`从同一历史数据生成，不手填分数。
更早k=2的clean-room先导实验在附录单列，明确其非PPO损失和不同师生身份。

附录A以“实验范围与拓展分析”组织叙述，说明完整Block3更新的比较对象与归因范围，
合并历史附录中重复的未完成实验清单。单seed、区间定义、数据覆盖和基线范围等事实保留，
不将整体更新收益写成仅由advantage平均造成，也不更改任何实验数值。

## 构建

在仓库根目录运行：

```bash
bash scripts/build_iclr2027_paper.sh
```

需要Tectonic、静态Noto Serif CJK SC/Noto Sans CJK SC字体，以及Poppler的pdftotext/pdfinfo。
本机同名SC可变字体会触发旧XeTeX字体索引错误，因此使用Noto官方CJK仓库的静态OTF版本。
默认Tectonic路径为`$HOME/.cache/opd-paper-tools/bin/tectonic`，可用`TECTONIC`变量覆盖。
产物为`build/en/main_en.pdf`和`build/zh/main_zh.pdf`。
检查器验证英文正文页数、引用、AI声明、常见匿名泄漏和未完成标记，但不代替人工科学审阅。
默认还要求4B Base和新增指令版两组对照的全部权重、4项benchmark均已通过验收，
并验证指令版初始学生及四个checkpoint的配对题目区间与汇总分数一致。
评测尚未完成时可用`bash scripts/build_iclr2027_paper.sh --draft`检查排版，产物不视为定稿。

## 提交前

正文上限9页；附录放参考文献之后。保留官方模板，不调页边距或字号规避页数。
本次历史Block3 Mean包含advantage、联合PPO ratio和归一化三项变化，不写成纯advantage对照。
逐benchmark报告，不能用宏平均掩盖负结果；单训练seed不能写成多seed复现。
内部证据文件不公开，原始数据和模型留在实验存储。
作者还需复核全部结论、引用、AI使用披露、许可和匿名性；本工具不会自动提交OpenReview。

## 图表与来源

`scripts/build_qwen4_paper_assets.py`仅从已验收的真实产物导出小型汇总、表格和曲线。
评测刷新使用`export --no-diagnostics --question-steps 50 100 150 200`；
所有位置诊断由两组各200个NPZ导出，不靠checkpoint之间插值。
大体积未压缩诊断缓存只保留本地，不放入投稿包；原始轨迹仍保留在实验存储。
热图灰色表示缺失或覆盖不足，不代表零。每128-token格按有效观测求均值，
阈值看格内最大的原始单位置有效计数是否达到8，而非格内计数之和。

历史表格由`build_history_paper_tables.py`生成；历史曲线由
`build_history_paper_figures.py`从`data/history_public.json`生成。
这些工具不合并不同grader视图，不把缺失值改成零，不汇总四项benchmark为总分。
数值图全由数据程序化绘制；两张概念图的AI生成来源见内部图审计。

`scripts/build_qwen17_paper_assets.py`从已验收的本地指令版归档读取36份逐题评分文件，
核对SHA-256、模型revision、prompt协议、采样配置及两组6,400条训练输入对齐记录。
它不重新生成回答或修改grader，复用现有逐题bootstrap实现，执行10,000次配对重采样
（bootstrap seed20260923）。这些区间以一个训练seed和已有8次响应为条件，
不是多训练seed置信区间，也没有做多重比较校正。生成器拒绝覆盖已有产物目录。
复现时在包含numpy和matplotlib的隔离环境运行：

```bash
python scripts/build_qwen17_paper_assets.py \
  --snapshot LOCAL_ACCEPTED_BACKUP \
  --source results/qwen17_instruct_20260922/final_20260922_2316/results.json \
  --tables NEW_TABLE_DIRECTORY --figures NEW_FIGURE_DIRECTORY
```

测试绘图工具可使用隔离环境，避免修改训练环境：

```bash
uv run --no-project --python 3.11 --with pytest --with numpy --with matplotlib \
  python -m pytest tests/test_qwen4_paper_assets.py tests/test_qwen17_paper_assets.py \
  tests/test_history_paper_tables.py tests/test_history_paper_figures.py \
  tests/test_iclr2027_paper_check.py tests/test_paper_package.py \
  tests/test_overlap_sensitivity.py -q
```

`build_overlap_sensitivity.py`对最终已评分问题做离线敏感性分析，剔除内部数据审计的
四个字符串重叠ID，产生`generated/qwen4_overlap_final/`。不重新评分、不修改原分数、
不占用GPU，也不声称完成语义去污染。完整500题仍为主报告。

## 匿名源码包

完成严格验收后，使用`package_iclr2027_paper.py --paper paper/iclr2027 --output NEW.zip`。
程序采用文件白名单，只包含论文源码、官方样式、表格和图，不包含`internal/`、
服务器路径映射、原始响应、训练数据或权重。输出已有时拒绝覆盖。
源码包是LaTeX编译材料，不等于完整实验代码匿名发布；不能将整个内部目录直接作为supplement上传。
