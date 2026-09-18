# 历史1.7B重评完成与Llama启动故障恢复

## 当前结果

六个历史Qwen3-1.7B权重均完成全量四任务n8评测与存档验收，共30864条生成。
Step200结果如下，均为百分比，差值为Block3减Token的百分点。

| Benchmark | Token Avg@8 | Block3 Avg@8 | 差值pp | Token Pass@8 | Block3 Pass@8 |
|---|---:|---:|---:|---:|---:|
| MATH500 | 45.925 | 54.150 | +8.225 | 83.000 | 84.000 |
| AIME24 | 6.667 | 7.083 | +0.417 | 16.667 | 26.667 |
| AIME25 | 3.750 | 5.833 | +2.083 | 10.000 | 16.667 |
| AMC23 | 23.343 | 29.217 | +5.873 | 60.241 | 65.060 |

这是单训练seed、固定评测配置的点估计，不是跨seed显著性证明。Step50和100
也已完整保留，不只报告Step200。原始结果与比较表在v2目录的`qwen17/`。

## 故障边界

- v2队列在2026-09-18 23:04:54北京时间退出，控制器PID434226。
- 六权重重评、Llama初始Instruct学生完整评测、师生GPU提示检查已通过。
- Token第一个恢复探针在`ray.init`创建socket时失败：
  `AF_UNIX path length cannot exceed 107 bytes`。
- 原训练临时路径为`/limx_embap/tos/h17/0918v2/llama/train/tmp`。叠加Ray
  session时间戳、PID和socket文件名后超出Linux限制。
- 这是本次控制器路径配置错误，不是OOM或算法collapse。未产生任何Llama
  训练checkpoint或训练rollout；正式Token和Block3均未开始。
- ANTLR版本告警并非本次退出的直接原因；不为处理路径问题调整依赖版本。

## 经检查后的恢复方案

1. 保留v2完整失败现场和所有评测，不重启/覆盖旧队列。
2. 新控制器`recover_historical17_llama.py`，新目录
   `runs/20260918v4_llama32_historical17_recovery_seed21_ml2`。
3. 缓存缩短为`/limx_embap/tos/lh/r1`，实际训练TMPDIR为其`train/tmp`。
   为Ray时间戳和最大Linux PID保留长度预算，并在ml2真实启动CPU Ray worker。
4. 训练/评测仍调用冻结运行时`94be7ea1d659309256c8356681925bb9710895c4`。
   不更改模型、数据、seed21、原生Llama模板、历史think训练指令、评测提示、
   loss、batch、学习率、checkpoint策略或过程指标。
5. 对六权重和Llama初始评测重新校验原始结果、历史grader、轨迹存档与hash，
   复用初始评测；不重新生成，不调用GPU重复评测。
6. 两组均从原始ModelScope 1B-Instruct初始化，Teacher保持3B-Instruct。
   先Token两步断点恢复探针，再formal200和Step200/100/50完整评测；之后Block3。
7. 新旧队列均加锁，失败停止，没有无限自动重试或删除产物。

## 验证与启动

- 服务器Ray原生路径校验已复现旧路径失败、短路径通过。
- 本地完整测试662 passed、2 skipped，用时20.96秒；新增测试覆盖路径预算、
  仅允许已定位的启动故障恢复、拒绝已有训练产物、复用初始评测和固定训练运行时。
- v3恢复入口在23:39:13启动(PID621354)，但共享文件系统拒绝对只读描述符
  执行独占flock，报`EBADF`。失败发生在创建manifest/Ray启动前，无GPU任务。
  v3启动日志完整保留，不能将其当作已开始的训练。
- 已在ml2实测同一锁文件：`open('r')`失败、`open('a')`成功，文件内容保持
  不变。v4改用非截断的可写描述符，增加NFS语义回归测试。
- 锁修正后的完整测试663 passed、2 skipped，用时20.29秒；独立审查无P1/P2。
- 短路径真实CPU Ray gate已在ml2通过：成功启动本地Ray实例并执行remote
  worker，返回`passed=true`，最大PID的socket路径预算102字节。
- v4于23:43:15北京时间启动，controller PID623151，控制器部署
  `b00ad93385ef367cc88c499d3c89d9feaa515327`；训练/评测运行时仍为94be7ea。
  两份部署哈希检查均通过，23:43:23进入队列内Ray gate(PID623328)。
- 最终ml2针对性回归21项通过。此时正式训练尚未开始；读取v4的
  `queue_state.json`和`logs/controller.log`确认后续状态，不重复启动。
