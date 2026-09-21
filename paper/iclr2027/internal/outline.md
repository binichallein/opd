# 论文结构与论断边界

## 主线

研究问题不是“block必然更强”，而是相邻token共享蒸馏信号何时有帮助、何时伤害，
以及实现中ratio与归一化耦合如何影响这种解释。以最新严格配对Qwen4为主试验，
用全部历史数据展示跨模型/协议的不一致与数值正常但行为坍塌的反例。

## 主文安排

1. 摘要：问题、真实实现、主要数值、边界；最终数字在四checkpoint验收后填入。
2. 引言：on-policy状态分布与token反馈；局部共享的动机及未被隔离的实现因素。
3. 相关工作：GKD/MiniLLM、Revisiting/Rethinking、blockwise drift gating与窗口方法。
4. 方法与分析：sampled-token基线、Block3 mean联合ratio、梯度尺度与cross-credit展开，
   降方差的充分假设/偏差代价，不作普适有效性证明。
5. 实验协议：按版本分组，模型身份/训练池/prompt/seed/rollout/eval/grader完全明确。
6. 结果：逐benchmark主端点、checkpoint曲线、跨师生正负证据、梯度/熵/截断诊断。
7. 讨论与限制：没有三因素分离对照、单seed、格式判据与因果解释限制。
8. 结论、复现/伦理/AI声明。

## 附录

完整历史实验目录及结果表；早期seed7 pilot/weight ablations单列；blocksize sweep；
不同师生与窗口；旧prompt与新completion差异、seed及重复生成诊断；全部checkpoint；
数学推导细节；指标精确定义；未完成/中止尝试与证据缺口；匿名复现配置。

## 图表

主图：方法信用分配示意、最新四benchmark曲线、最新训练指标及位置热图。
附录：跨模型对照、block10数值/行为退化、全部系列紧凑表。
所有结果图使用实际摘要/诊断数据生成；概念图不冒充测量结果。

## 不允许的结论

不声称减少teacher前向次数、严格无偏full-vocabulary block KL、理论保证涨点、
普遍降低format error、跨seed稳健、SOTA、未运行的baseline胜出或prompt修复已因果证明。
缺失结果不记零；不同协议不混表求均分；失败历史不得删掉。
