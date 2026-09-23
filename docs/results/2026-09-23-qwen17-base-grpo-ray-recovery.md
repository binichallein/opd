# Base1.7B/GRPO4B: Ray启动等待故障

## 实际进度

- v4的Block3两步保存/恢复探针已通过，四rank模型、Adam状态、调度器、数据进度与RNG均已验收。
- 正式200步任务于北京时间20:00:36启动，但尚无第一步rollout或checkpoint。
- 20:14至20:19复核发现GPU空闲，主进程卡在`ray.init -> CoreWorker -> RegisterClient`的接收等待。
- 20:19:50旧队列记录failed。仅终止了已核实的阻塞训练进程组3170561，保留所有旧文件。

## 诊断依据

Ray 2.55.1预启动53个Python worker，其中PID3171231注册超时；随后52个worker空闲，
驱动注册未返回。native py-spy栈和Ray日志已封存于旧目录
`supervision/ray_stall_20260923T1220/`（实际抓取UTC12:19:05）。
这不是OOM、梯度异常或模型collapse；模型训练尚未执行。

[Ray 2.55.1 WorkerPool源码](https://github.com/ray-project/ray/blob/ray-2.55.1/src/ray/raylet/worker_pool.cc)
的`ExecuteOnPrestartWorkersStarted`会等待预启动worker数达到门槛后再返回驱动注册。
现场症状与该等待条件一致；单个worker首次注册超时的底层诱因尚未确定。

## 最小修复

新建v5独立目录，不覆盖v4，仅设置Ray启动环境：

```text
RAY_prestart_worker_first_driver=0
RAY_enable_worker_prestart=0
```

保持64个CPU资源、4张GPU、所有训练超参数、数据、prompt、采样seed、模型及历史loss不变。
worker按需启动，不修改Ray安装包。新64-CPU/0-GPU启动验收有360秒上限，必须执行真实远端任务。

恢复控制器：`scripts/recover_qwen17_base_grpo.py`。
复用原Block3恢复验收前，重新核对全部受保护输入、训练脚本字节、rank状态、优化器哈希及原始轨迹。
正式Block3从原始学生重新初始化，绝不把probe Step2当作正式初始化。Token仍执行自身保存/恢复探针。
队列顺序仍为Block3训练/200、150、100、50完整评测，初始学生完整评测，再Token训练与同四步评测。
每组保留200步轨迹、逐步诊断和四个完整checkpoint；不删除旧产物。

新目录：`runs/20260923v5_qwen17_base_grpo_blockfirst_seed21_ml2`。
这是恢复方案记录；是否已经恢复训练，以新目录实时日志和rollout为准，不以脚本存在代替成功。

## 部署记录

- 新增7项回归覆盖配置不变、旧故障验证、全部rank验收和原生SHA清单解析；先失败后实现。
- 本地相关回归211 passed；ml2不可变部署核心回归80 passed。
- 代码`be2eff65325ed35d26911046beb4ab42d82bf8ad`已推送GitHub并校验部署。
- 20:26:13北京时间nohup启动v5，PID/SID3182452、PPID1，当前先执行输入与原验收的复核。
- 本地故障证据及两步probe的轨迹/诊断备份位于
  `/home/tyf/paper/outputs/qwen17_base_grpo_pair_20260923/ray_recovery_2030/`。
  目录名是操作标签，不是精确启动时间；不包含大权重，完整权重仍在ml2。

## 20:48正式训练复核

64-CPU Ray验收26.73秒通过。正式任务20:30:28启动，PID3183928，
驱动已越过原卡点，完成数据索引、模型加载、vLLM预热及连续4次真实更新。

| Step | 截断条数/32 | 裁剪前grad norm | PG loss | 学生熵 | 教师熵 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 2.6318 | 0.07384 | 0.20472 | 0.17354 |
| 2 | 3 | 2.5178 | 0.07068 | 0.14860 | 0.12701 |
| 3 | 0 | 2.5563 | 0.08829 | 0.36013 | 0.29457 |
| 4 | 1 | 2.7442 | 0.09167 | 0.27980 | 0.24169 |

截至该时点，128条正式轨迹、4份位置NPZ及逐步标量已生成。
累计截断5/128=3.90625%，think标签为0，标量无NaN/Inf；日志没有OOM或异常traceback。
前两步各有1条周期性重复尾段，不能把“数值有限”写成“全部输出正常”。
只有早期训练证据，不宣称200步稳定、教师筛选有效或Block3涨分。

`perf/max_memory_*_gb`在实际代码中除以1024^3，单位是GiB，不是十进制GB。
其中allocator reserved计数出现超过80GiB物理容量的记录，不能据此称GPU实际占用了85GB
或已经OOM。该计数含义需另行核对allocator行为；物理显存占用应以独立NVML采样为准。
原日志保留，未调整训练参数或修改该计数实现。

完整两步审计与15文件本地快照校验已通过，路径为上述本地目录下`formal_step000002/`。
后续监督快照继续保留在新运行目录`supervision/`中。队列仍按原计划自动训练和完整评测。

20:49:40补充：前4步128条轨迹的源题顺序、prompt token、请求seed、EOS/mask/logprob
完整审计通过，周期性重复尾段为3/128。四步快照为`formal_step000004/`，
对应[结构化监督记录](../../results/qwen17_base_grpo_pair_20260923/supervision_step4.json)。
