# Token 完整评测与双语论文

## 授权与顺序

用户要求立即评测，Step200优先，然后据全部已有实验撰写ICLR2027中英文论文。
仅访问ml2。Token训练已完成，四权重验收通过；200步、每组6400条训练输入
完成成对核验。新评测目录为`20260921v1_qwen4_completion_token_eval_seed21_ml2`，
顺序200、150、100、50。保留旧训练及Block3评测不动，不启动新训练。

## 固定评测协议

复用Block3评测入口，仅用不可变EvaluationSpec选择Token源权重与独立输出。
推理和合并使用冻结训练发布`0f9161f02f08287fb07f0375ad0a6bda81133ff0`。
MATH500(500)、AIME24(30)、AIME25(30)、AMC23(83)，每题8次，独立seed21至28，
temperature1.0、top_p0.9、max_tokens16384。4卡各TP1，vLLM显存比例0.9。
显式`qwen3_completion_boxed_v1`，不使用ChatML，thinking=false，EOS151643。
历史grader SHA256 `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
逐benchmark报告Avg@8、Pass@8、缺失boxed、引擎length停止率；不汇总成总分。
每权重5144条原始轨迹、32份归档，检查原始输入token、数据/模型hash和计分一致性。

## 论文证据与边界

主比较使用匹配协议的Token与历史Block3 Mean，Step200为主端点，其他节点呈现曲线。
历史Block3不仅改变advantage，还改变joint PPO ratio和loss归一化，必须如实表述。
全部历史系列建立证据目录：先导、不同师生、窗口、坍塌、提示诊断、中止/失败尝试。
不同prompt/grader/seed/训练池不能混作严格方法对照；缺少原始证据的项目标明缺口。
单训练seed和小样本AIME限制必须说明，问题级bootstrap不能替代跨训练seed稳健性。

英文使用官方2027匿名模板；中文为对应阅读稿。正文遵守九页上限，完整历史表置附录。
核验官方AuthorGuidelines和AIPolicyForAuthors，披露AI参与研究设计、代码、分析、写作、翻译。
结果未完成前不写最终获益结论；不自动提交投稿。
