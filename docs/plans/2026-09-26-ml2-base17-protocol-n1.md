# ml2：Base1.7B / GRPO4B 新协议对照

## 授权与范围

用户批准在ml2执行此前建议的协议对照；ACP消融继续运行、不改动。
教师为原公开Qwen3-4B-Base-GRPO，学生为原Qwen3-1.7B-Base，复用历史模型字节。
对照为已完成的 `20260925v4_historical17_pair_n1_step100_save25_seed21_ml2`。
不覆盖历史结果，不把这次新协议与另一台机器的指令模型实验做因果比较。

## 预先固定的配置

- 只改变生成协议：Base裸题目、boxed指令与Solution:，不使用ChatML或think标签；
  训练评测均用 `qwen3_completion_boxed_v1`。原生EOS151643，无额外chat停止符；
  按首次EOS保留有效response，真实长度、padding、mask和logprob逐条存档及检查。
- 原每请求seed21不变，数据seed21；100步、32题x1条、PPO minibatch32、epoch1、lr2e-6。
  相同DAPO文件、同3200个物理行位置、相同四benchmark、同历史grader。
- 完整历史Block3 Mean先训再评测，之后Token OPD独立从原学生初始化训练再评测。
  不运行ACP的adv3/scale3，不重评初始学生。每组保存25/50/75/100全部完整状态且不删除。
- 每组完整评测100/75/50/25，每题8条，MATH500/AIME24/AIME25/AMC23各自计分。
  保存全部训练/评测rollout、Step1及每5步指标和位置数据，训练后生成热力图。
- actor/ref micro1、rollout logprob micro4、vLLM0.6、响应上限16384、Ray64CPU，和旧ml2配置相同。
- 每组先保存1步、退出、恢复至2步，使用100步scheduler；正式训练不继承探针更新。
  服务端nohup运行，任何失败保留现场并停止队列，不静默调参或自动重跑。

## 解释边界

这次验证的是完整输入/终止协议口径，不声称单独隔离了提示文本的因果效应。
保留旧实现EOS mask算法，只显式校验它与实际生成长度一致；不暗中换loss或随机种子。
完整报告各checkpoint与各benchmark，Step100为主结果；单训练seed不能证明稳定显著提升。
教师此前的能力筛选仍是inconclusive，用户已授权训练；不将这次工程验收说成能力筛选通过。

## 执行状态

- [x] 确认历史建议、用户授权、ml2空闲与原对照完成。
- [ ] 协议入口、失败测试、回归与配置/数据核对。
- [ ] 不可变部署、GPU保存恢复与正式首步。
- [ ] 两组完整训练、八次完整评测与归档。
