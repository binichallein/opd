# 0.6B Block3完整评测启动记录

用户明确要求直接启动评测。本记录是启动证据，不是完整评测结果。

## 身份与范围

- 仅ml2，新运行`20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2`。
- 控制器版本`831c0af58f34ad90ac6776f1a5444622a5b82262`，
  `scripts/run_nonthinking_block3_eval.py`，PID252025。
- 2026-09-17 22:15:27北京时间通过nohup启动，实测PPID=1，脱离SSH会话。
- 评测器、权重merger仍固定在`0ce73aa42d7f734b9d34c379f474458bb6d48771`，
  不使用新版本修改解码或grader。后续文档提交不替代冻结版本。
- 来源为v1的Block3完整checkpoint，固定顺序200、100、50。Token来源为v1
  已验收的同step评测，复核后复用，不重新生成。没有新训练或旧队列重启。

## 配置与计分

MATH500 500题、AIME24 30题、AIME25 30题、AMC23 83题，每题8次生成，
seeds21至28，temperature1，top_p0.9，max_tokens16384，enable_thinking=false。
每个checkpoint5144条，三个共15432条；各benchmark分别报告Avg@8、Pass@8，
并给出同step的Block3减Token百分点差异，不合并总分。

历史grader SHA256：
`04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`。
新合并权重和结果仅写v2。退出异常、完整性/哈希异常即停，不自动重试或删产物。

## 启动验收

- 新来源选择及队列测试先失败再通过；本地针对性测试17通过，代码复查无阻塞项。
- 全量测试分环境验证：ml2 497 passed、2 skipped；依赖真实Git工作树的控制
  脚本测试在本地45 passed。共覆盖542项通过及2项跳过，不是一次单环境全绿。
  初始本机默认Python有torch iJIT链接错误；远端归档不含.git，故相应45项在
  本地Git工作树测试。没有为测试更换ml2实验依赖或修补冻结运行时代码。
- 两份运行时文件哈希通过，v1已完成及Block3完整checkpoint/轨迹验收已核实。
- `preflight.json`通过：源Block3身份、run_card配对、Token三个视图原始/判分
  一致性、643题完整覆盖、历史grader及证据哈希均通过检查。
- Step200合并任务于22:16:45启动，PID252118，目标`merged/block3_step200`。
  未向旧Block3或Token checkpoint写入HF权重。
- Step200评测于22:17:32启动，PID252473。实际eval_config包含四个任务、n8、
  seeds21至28、temperature1、top_p0.9、max_tokens16384、external grader和
  enable_thinking=false，与Token配置一致。
- 643题实际输入token哈希为
  `af456ee536c0c0f75eb23311ec94e33020c384074993a13ae302e6a11f57dac9`，
  与Token五个视图一致；最长818 token。模板中的已关闭空think控制前缀未删改。
- 22:20后确认四个引擎均完成初始化，四次Supported_tasks=generate，随后GPU
  利用率82%/84%/82%/85%、各78953MiB，已在MATH500实际生成，未见启动错误/OOM。
  该显存用量来自与Token相同的评测器预算0.9。结果尚未完整落盘，不报告分数。
- 沿用的tokenizer regex warning和grader SyntaxWarning仍存在，但输入token与
  grader哈希/正反例自检通过；未为消除警告在线改参数或更换grader。

## 后续入口

实时状态：`queue_state.json`、`controller.log`。
每个视图：`evaluations/block3_step{200,100,50}/`中的`eval_card.json`、
`prompt_contract.json`、`model_identity.json`、`logs/eval.log`、`outputs/`、
最终`acceptance.json`。
全部完成后生成`block3_comparison.json/.md`及`block3_eval_acceptance.json`。

训练时的严重输出退化仍是有效警报；启动评测不代表方法已被验证有效。
