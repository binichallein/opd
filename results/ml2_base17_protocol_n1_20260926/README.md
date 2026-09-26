# ml2 Base1.7B / GRPO4B 新生成协议对照

## 本轮实验

- 教师：原公开Qwen3-4B-Base-GRPO；学生：原官方Qwen3-1.7B-Base。
- 相同模型字节、DAPO数据文件、3200个物理行位置及顺序、seed21、每请求seed21。
- 每步32题，每题1条；100步，lr2e-6，PPO minibatch32/epoch1。
- 完整历史Block3 Mean和Token OPD，不是ACP的adv3/scale3消融。
- 新协议：裸题目、boxed指令、`Solution:\n`，不用ChatML或think标签；训推一致。
- 原生EOS151643、无额外chat停止符；保留旧EOS mask算法，校验其与真实长度一致。
- Block3训练后完整评测100/75/50/25，再从原学生独立初始化Token并重复流程。
- 不重评初始学生。每组完整保留25/50/75/100的模型、优化器、scheduler、RNG和采样状态。
- MATH500、AIME24、AIME25、AMC23分别计分，每次643题x8；使用历史grader。
- 保存全部训练/评测rollout、Step1和每5步诊断及位置数据，训练后生成热力图。

## 对照与解释边界

旧对照为ml2已完成的 `20260925v4_historical17_pair_n1_step100_save25_seed21_ml2`。
模型、数据、loss、环境关键版本和随机种子与它匹配，改变整个输入/终止协议；
不能把结果解释为单独prompt文本的因果效应。Step100为主结果，同时报告其他三个checkpoint。
仅一个训练seed，不声称统计显著或普遍有效。教师此前筛选仍为inconclusive，用户明确授权训练。
ACP上的指令版loss消融不受影响，也不作为本轮协议变化的直接因果对照。

## 启动记录

- 不可变代码：`3316918c4ef5488d4ca4b5100c62b831eede979d`。
- 服务端根目录：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd`。
- 新run：`runs/20260926v1_qwen17_base_grpo_protocol_n1_seed21_ml2`。
- 北京时间2026-09-26 16:31，以nohup/setsid启动controller4142950；PPID1已验证。
- 119项本地回归通过，250项ml2运行时回归通过。一个依赖Git数据库的历史检查在本地单独通过。
- 707题的真实collector输入检查通过；包括64题筛选集及643题benchmark。
- 首个预检因历史模型目录的`.gitattributes`与官方清单不同而停止，未启动GPU训练。
  修复只排除未被加载的仓库说明文件；所有权重、模型配置、tokenizer和历史artifact清单仍受校验。
  失败日志与修复后验收均保留在`preparation/`，没有替换旧模型。
- 每组必须先通过自己的保存1步/退出/恢复2步GPU探针；正式训练始终从原学生开始。
  失败停止并保留现场，不自动重试或静默修改参数。

## 17:23 启动验收

Block3的保存1步/退出/恢复2步验收通过，四rank模型、Adam状态、scheduler、RNG和数据进度
均核对，恢复后原第1步文件哈希复核通过。两次探针正常退出，64条原始轨迹通过审核。
探针截断分别1/32、3/32，周期性尾部分别1/32、2/32，think标签均为0。
清理阶段出现DataLoader worker的Killed警告，但不应将其当成更新失败：两次退出码均为0，
完整状态、优化器和实际恢复推进检查均通过。

正式Block3于北京时间17:06:22从原学生启动，PID4169228，`resume_mode=disable`。
17:23:40验收已完成的前5步，所有160条完整轨迹均符合实际prompt、sampling、原生EOS、
padding/mask与logprob契约，并匹配冻结的物理行计划及旧两组的题目顺序。
实际输入token按新协议变化，不能声称与旧协议的prompt token相同。

| Step | 正式截断/32 | 周期性尾部/32 | Grad norm | 旧Block3截断/32 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 16.617 | 16 |
| 2 | 1 | 0 | 3.134 | 17 |
| 3 | 0 | 0 | 3.012 | 13 |
| 4 | 2 | 1 | 2.186 | 10 |
| 5 | 2 | 0 | 2.469 | 8 |

新协议合计6/160截断（3.75%），旧Block3相同前五批为64/160（40%）。这不是最终benchmark
结果，不能据此宣布Block3优于Token；新Token组尚未启动。重复/截断也没有完全消失，首批的
周期性加法输出发生于首次更新前。周期性尾部是现有检测器的统计，不等于穷尽所有语义重复。

Step1和Step5的标量及全部12类16K位置数组通过完整性、覆盖和有限数校验，非有限数、溢出、
下溢计数均为0。学生entropy为0.2363/0.1670，教师entropy为0.2040/0.1444，Top16 overlap为
0.7216/0.6854，sign-flip rate为14.65%/14.80%。这些是不同训练批次上的观测，不能直接当作
同一固定测试集的趋势。热力图所需位置数据已经保存在服务器，训练结束自动生成完整图。

首批正式32条输出token与探针一致，但实际更新有小数值差异，第二批生成也不同；只保证保存
和恢复完整训练状态，不宣称vLLM或优化器逐比特重演。原teacher能力筛选仍为inconclusive。
另存3秒间隔的nvidia-smi显存观察（84个时点，最大观测52694MiB）；它不是连续真实峰值，
也不能将日志中的allocator reserved计数直接当作驱动层物理显存占用。启动5步未发生OOM。

轻量证据：`probes/block3_mean/`（恢复/状态/轨迹验收），`supervision/startup_acceptance.json`
（正式160条及Step1/5检查），`supervision/gpu_memory_observation.json`（只读采样记录）。
监督期间没有重启训练、修改参数/loss、热改部署或接触ACP。后续仍按Block3训练100步、完整评测
100/75/50/25、独立Token保存恢复/训练/完整评测执行；不重评初始学生，不删除任何checkpoint。

本目录保存轻量审计证据，不是完整模型备份。原始rollout、完整checkpoint与运行日志保留在ml2
上述持久化run目录。启动快照不是实时状态，报告进度前应读取服务器的`queue_state.json`和训练日志。
