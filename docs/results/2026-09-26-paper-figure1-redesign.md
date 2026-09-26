# 论文图1改为方法图

用户指出旧图不能直观看出本文工作，并提供Self-Distilled Reasoner Figure1作为组织方式参考。
本次仅重绘图1、修改中英文图注，没有改变loss实现、实验数据、结果或正文结论。

## 图中结构

1. 学生生成on-policy轨迹，冻结教师在同一前缀上打分，形成师生log-prob差值。
2. Token OPD保留各token独立反馈；Block3对连续三个token的反馈求均值并在块内共享。
3. 右侧展示token ratio与联合block ratio，以及对应的归一化目标，梯度仅流向当前学生。

明确保留历史实现同时改变共享反馈、联合ratio和loss reduction的事实。
上下两条路径代表独立训练对照，不是两个目标相加；先生成完整轨迹，不是每三token更新。
参考图的OPSD自教师/特权信息机制不属于本方法，未移植到新图。

## 生成与验证

使用内置OpenAI image_gen，并做两次局部修正：让反馈分支来自a_t而非未评分token，
删除边界处重叠标签。提示词规范在`paper/iclr2027/internal/figure1_redesign_20260926.md`。
图中没有真实轨迹、性能数字或效率提升主张。

双语PDF重新编译通过，英文正文9页、中文7页；无溢出框和未解析引用。
28项论文检查和结果资产回归通过；首页渲染经视觉检查。匿名源码包80个条目，完整性检查通过。
上述检查不是学术有效性或审稿结论。未上传OpenReview。

## 交付

新版路径：`/home/tyf/paper/outputs/iclr2027_20260926_figure1/`。
包含英文PDF、中文PDF、图1PNG、匿名源码ZIP及SHA-256清单；9月23日交付不覆盖。
Windows路径：`\\wsl.localhost\Ubuntu\home\tyf\paper\outputs\iclr2027_20260926_figure1`。
最近100步实验及正在运行的消融尚未纳入该PDF。

同日按用户反馈补齐Block3分支到当前学生的`gradient update`标签，仅局部调整右侧连线，
不重做图的结构。直接覆盖此交付目录中的图、双语PDF和源码包，并刷新SHA-256清单。
