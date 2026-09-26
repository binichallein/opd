# 图来源

`study_overview.png`与`credit_assignment_concept.png`为OpenAI image generation生成的概念图。
未使用实际响应、私有模型、作者身份或未经公布的实验数字作为图像服务输入。
前者展示rollout、teacher反馈、两个实现和学生更新；后者使用人为构造的
`(+2,-1,+2)`及均值`+1`说明符号翻转，不是实测数据。
原生成日期2026-09-21；图1于2026-09-26重新生成并替换为白底方法图，图2未修改。
图1参考Self-Distilled Reasoner (arXiv:2601.18734) Figure1的流程组织方式，
但没有复制其自教师或privileged-context机制：本研究使用独立冻结教师。
中心画出逐token独立反馈与三token均值共享的区别，右侧保留联合ratio及有效长度加权loss。
图注注明两种目标分开训练、反馈停止梯度、mask尾块以及完整轨迹采样后更新。
生成使用内置OpenAI image_gen，进行了四次针对性连线/标签修正；最终检查公式、token数量、
箭头来源、标签边界和论文缩放后的可读性。图中不包含性能数字或效率提升主张。
本次提示词规范记录在`internal/figure1_redesign_20260926.md`。
第三次修正直接补齐Block3到当前学生连线的`gradient update`标注，与Token分支对称，
只调整该连线路径以容纳文字。现有图、双语PDF及源码包原位更新，没有新增交付版本目录。
第四次按用户要求将两条更新支路汇合为一根标注`gradient update`的共享箭头；
汇合点不是loss求和，图注仍明确两行代表独立训练方案。
所有性能曲线、训练曲线、热图与表必须从实际数据程序化生成，不使用图像模型造数。

scientific-schematics脚本需要未配置的OpenRouter凭证，概念图改用当前可用图像生成工具。
不调用未知外部API密钥，也不将AI图像声称为测量证据。
