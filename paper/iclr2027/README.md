# ICLR 2027双语论文工作目录

题目：Block-Level Credit Assignment in On-Policy Distillation。
英文为按ICLR2027匿名样式编排的论文稿；中文为对应阅读稿，不是另一次投稿。
所有实验结论须以验收完成的数据为准。不能把本目录的产生本身视为实验验证或可直接投稿的保证。

当前四个Token权重已全部验收，最终数据为`data/qwen4_final_20260921.json`，
表格为`generated/qwen4_publication/`。`qwen4_final/`为相同分数的早期标签版本，
publication版本仅规范模型名。旧partial缓存不用于成稿。
全文保留历史正向、负向、中止与先导实验；完整性不等于所有实验具有相同证据强度。

## 结构

- `main_en.tex`、`main_zh.tex`：两个入口，共用官方2027模板及同源结果。
- `content_*`、`method_*`、`results_*`、`appendix_*`：正文、推导、结果和附录。
- `references.bib`：核验后的参考文献。
- `data/`、`generated/`、`figures/`：紧凑证据、自动表格和图；不存完整rollout或权重。
- `internal/`：内部来源映射、证据审计、合规记录。**不得放入匿名投稿包。**
- `build/`：编译中间件、PDF、文本与验收结果，不入Git。

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
默认还要求当前两方法的4个权重、4项benchmark均已通过验收，并有Step200配对题目区间。
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

测试绘图工具可使用隔离环境，避免修改训练环境：

```bash
uv run --no-project --python 3.11 --with pytest --with numpy --with matplotlib \
  python -m pytest tests/test_qwen4_paper_assets.py \
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
