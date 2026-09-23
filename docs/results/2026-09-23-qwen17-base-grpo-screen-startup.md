# 4B-GRPO到1.7B-Base教师能力测试启动记录

本记录为2026-09-23启动时点证据，不是最终成绩或教师通过结论。
用户最新要求为“先只做教师能力测试，训练另定”。本队列没有训练任务，测试通过也不会自动训练。

## 身份与协议

- 学生：官方`Qwen/Qwen3-1.7B-Base`，ModelScope revision `b0786a09cd6ee101cd8c90e30a5727beb8230544`。
- 教师：历史公开`lllyx/Qwen3-4B-Base-GRPO`，不是Qwen官方GRPO版本或用户私有模型。
  复用已核验文件哈希及本地下载revision证据 `1f3b2966edfb75f2f98a00617588c1f748088422`；
  此revision证据不应写作历史下载时已经显式固定revision。已有文件只读校验，不重新下载。
- 主视图为Base裸题目、boxed指令及`Solution:`，不使用ChatML或think标签。
  教师接收学生原始prompt token IDs及原始续写前缀，主视图只使用原生EOS151643。
- 教师原生chat视图另作对照，显式关闭thinking，采用停止集合[151645,151643]。
  它同时改变输入及停止策略，不是只改变模板的因果消融。
- 64道既有冻结DAPO题，独立答题每题2次；前32题取学生sample0的因果前缀，双方各续写2次。
  加教师原生独立答题对照，共512条正式输出；另有两模型各4条256-token短检查。
- 正式总响应预算16384，温度1、top_p=0.9、top_k=-1；续写扣除已有前缀长度。
  复用历史grader、按题目/样本/阶段派生的seed和原有验收门槛。
- 这不是完整benchmark评测，也不是新未见题测试集；GRPO教师训练用过DAPO。
  能力筛选通过不等于已经证明OPD训练有效。

详细协议和预设门槛见[实施计划](../plans/2026-09-23-qwen17-base-grpo-teacher-screen.md)。

## 启动与核验

所有时间为北京时间。只访问ml2，未访问train。

- 18:33:34：服务端nohup启动。PID/SID均为`3138448`，PPID为`1`，不依赖本地SSH持续连接。
- 不可变运行代码：`91fe84ea82d62d3cbaf6a2843e3639bd018c15bb`。
  部署位于远程仓库根目录的`analysis_deployments/<commit>`，文件清单校验通过。
- 新旧筛选器、资产及控制器共173项CPU测试在本地和ml2均通过。
  ml2测试有一条pytest缓存跨设备写入警告，不影响测试结果或推理输出。
- 两个短检查单元均exit 0，各4条原始输出完成封存，生成think标签均为0。
  两组均4/4达到256-token短上限，不能据此推断16K正式测试的截断率。
- 18:39:45：学生独立答题已落盘13/128条，教师学生输入视图1/128条；
  教师原生视图引擎初始化完成，尚未落盘首条。GPU0/1/2利用率分别46%/49%/55%，GPU3空闲。
  续写单元等待学生独立答题完成后自动调度。
- `queue_state.json`为`running`且`training_started=false`。
  最终状态、有效boxed缺失、截断、重复及正确率须等待全部输出封存和汇总核验。

## 证据位置

远程仓库根目录：

```text
/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd
```

本轮目录为`runs/20260923v3_qwen17_base_grpo_teacher_screen_ml2`，包含
`startup_launch.json`、`queue_process.json`、`queue_manifest.json`、`queue.log`、
每个任务的命令和日志，以及`qualification/cells/`下的请求、原始输出和评分。
全部正式轨迹会在服务器保留；目前本地仅是启动快照，不是最终全量归档。

本地启动快照：

```text
/home/tyf/paper/outputs/qwen17_base_grpo_screen_20260923/startup_1840
```

包含准备清单、固定题目、启动状态、任务命令，以及两模型已完成的短检查完整轨迹。
两组短检查各5个完成文件的SHA256均验证通过；大批原始轨迹不提交Git。

| 文件 | SHA256 |
| --- | --- |
| `queue_manifest.json` | `2e00a10b7223f60fa59581bbfa707c08bcb0a27613f3763ad94219c54d9bfac6` |
| `qualification/prepare_manifest.json` | `d6f5e5a7aa1a6750d80d7367e99827550dc77ae81bfaf94f12fe1ee89508775f` |

最终报告尚未生成。无论最终通过、未通过或证据不足，本队列均在能力测试后结束，训练另定。
