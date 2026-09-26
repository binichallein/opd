# 图1方法图重绘

## 目的

旧图强调审计和评测，没有直接展示本文对OPD做了什么改动。新图以方法的信息流为主体，
参考用户提供的Self-Distilled Reasoner (https://arxiv.org/pdf/2601.18734) Figure1的
左到右流程组织，但保留本文独立教师、sampled-token反馈和历史Block3实现的真实定义。
不增加实验结论，不把正在进行的消融写成已完成。

## 最终生成规范

使用内置OpenAI image_gen。宽幅白底论文方法图，三个编号区域，清晰黑色数学标签，
青色Token基线、珊瑚色Block3、浅蓝模型模块。禁止性能数字和效率提升口号。

1. 左侧：问题x -> rollout student pi_old -> 学生生成的y1/y2/y3/... -> 冻结teacher
   在相同prefix打分 -> a_t = log pi_T(y_t|h_t) - log pi_old(y_t|h_t)。
   从a_t公式框引出分支，不能从未评分的token串直接引出反馈。
2. 中间上下对照：Token OPD三份a分别指向对应token；Block3把连续三份a合成均值，
   再用三条箭头把同一均值共享给三个token。明确baseline/ours。
3. 右侧：Token使用token ratio及token-normalized clipped surrogate；Block3使用
   R_j = product(r_t)和-(1/T)sum_j n_j C(R_j, mean_a_j)。两条独立训练路径都只更新pi_theta。
4. 底部说明C是clipped PPO surrogate、n_j是block有效token数、梯度仅更新当前学生。
   不表示每三个token进行一次优化，不表示教师非自回归打分或枚举完整joint KL。

第一次生成后修正了左侧信号来源和教师到a_t的连线；第二次局部修正删除了边界处多余的
Per-token feedback标签。保留所有数学符号及其他结构。最终图使用同一英文数学标记，
中英文正文各自提供对应图注；旧交付PDF不覆盖，新PDF放入20260926图1更新目录。

用户随后指出Block3到学生的连线缺少与Token分支对应的标注。第三次局部修改只将
该短箭头改为从block loss右侧进入学生框底部的折线，补上相同的`gradient update`，
避免文字重叠；其余布局和公式不变。20260926目录中的现有图、PDF和源码包原位覆盖更新。
最终按用户建议简化为两条loss支路汇合、一个连接点、一根指向当前学生的箭头，
只保留一次`gradient update`标注；不使用加号或求和节点。继续原位更新相同交付文件。
随后修复合并编辑中遗漏的Block3到block目标函数框的红色水平箭头，闭合左侧浅蓝外框
右边界，将黑色反馈干线移至框外间隙。确认上方Token连线和右侧汇合更新箭头未改变。
用户进一步确认要修的是两个小蓝色模型框，而非外层大框。此次提示词仅要求增强
Rollout student与Frozen teacher的右侧竖边并保留现有圆角，其他内容不变。
从image_gen候选图中仅合成两个边缘条带到原图（左上角坐标，右下界不含）：
`[444,447) x [181,219)`和`[444,447) x [437,483)`。共252像素变化，条带外逐像素一致；
两条竖边分别38/38和46/46行均保持连续蓝色。原位更新同名图、双语PDF及源码包。

## 科学含义检查

- 教师不产生学生的rollout；无privileged information输入。
- pi_old固定，a_t停止梯度；pi_theta通过ratio收到梯度。
- 三token均值示例仅表示完整块，尾块mask及n_j由正文和图注说明。
- 完整历史Block3不只是均值共享，还改变ratio及reduction。
- 图中两条学习目标不是同一次训练相加，不是实测性能证据。
