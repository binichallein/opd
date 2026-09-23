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
