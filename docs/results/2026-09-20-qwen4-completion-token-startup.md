# Qwen4 Completion Token 等待队列启动记录

2026-09-20 14:34:46北京时间，已在ml2启动独立服务端nohup控制器，PID1529721。
14:35:04实测PPID=1、SID=1529721，SSH断开不影响等待/训练。

## 当前状态

- `waiting_for_evaluation`，尚无Token `train.pid`；没有开始GPU训练。
- 前置Block3 Step50评测PID1523774仍在运行，未暂停、修改或重启。
- 四个Block3权重均完成并通过全量验收后，检查GPU空闲和公共配置，再自动启动Token。
- 任一前置/配置检查失败则停止并保留现场，不自动重试或覆盖。
- 此快照不等于Token已开训；后续必须查询独立队列实时状态，不得重复启动。

## 版本与产物

- 控制代码：`7131bc181c0bcddb6f5f95d3c5d379297c361b79`，已同步GitHub。
- 训练版本不变：`0f9161f02f08287fb07f0375ad0a6bda81133ff0`。
- 远端控制器：`analysis_deployments/7131bc181c0bcddb6f5f95d3c5d379297c361b79/scripts/run_qwen4_completion_token.py`。
- 实验根目录：`/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd/runs/20260920v1_qwen4_completion_token_seed21_ml2`。
- 根目录保存 `launch.json`、`queue_manifest.json`、`queue_state.json`、`controller.log`；
  开训后子目录 `token_opd` 保存command、run card、日志、rollout、diagnostics、四个完整checkpoint。
- 同名JSON保留实际命令、PID、源码/部署hash；后续文档commit不替换运行版本。

## 验证范围

本地相关测试114项通过；另执行gate/pair测试27项，其中有重叠，不能相加宣称独立用例数。
独立静态代码复核未发现启动阻断问题。远端训练venv导入检查确认gate来自冻结版本，
实际历史Block3 run card与已接受Token probe可按唯一许可差异对齐。
正式command、模型/数据hash检查在GPU空闲后执行，不能把当前等待状态描述为训练预检全通过。

正式训练仍为seed21、200步、Step50/100/150/200完整保存、每步32条原始rollout和诊断，
从原始4B Base初始化。详见 `docs/plans/2026-09-20-qwen4-completion-token.md`。
