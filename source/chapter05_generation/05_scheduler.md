# 05.5 Scheduler 调度器

## 学习入口

Scheduler 把请求推进顺序变成可观察、可测试的规则。完成后应能追踪 waiting/running 请求，解释 round-robin 如何退出，并确认每个请求只调用自己的 Model Executor。

- 前置阅读：[`05.3 Naive Generation`](./03_naive_generation.md)、[`05.4 KV Cache`](./04_kv_cache.md)
- 源码直达：<a href="../_code/chapter05_generation/src/scheduler.cpp"><code>scheduler.cpp</code></a>、<a href="../_code/chapter05_generation/src/request.cpp"><code>request.cpp</code></a>、<a href="../_code/chapter05_generation/tests/chapter05_scheduler_test.cpp"><code>chapter05_scheduler_test.cpp</code></a>
- 相关阅读：[`05.6 Sampler`](./06_sampler.md)、[`05.9 Request 生命周期`](./09_request_lifecycle.md)、[`05.10 Continuous Batching`](./10_continuous_batching.md)

## Round-robin 基线

当前 `RoundRobinScheduler::run_with_stats()` 先 prefill 尚未建立 Context 的请求，再循环调用 `Runtime::decode_one()`。一轮没有任何请求产生进展时退出。

```python
def run_round_robin(runtime, scheduled_requests, stream_token):
    for item in scheduled_requests:
        if item.request is not None and item.executor is None:
            runtime.prefill(item.request)

    progress = True
    while progress:
        progress = False
        for item in scheduled_requests:
            request = item.request
            if request is None or item.sampler is None or request.terminal():
                continue

            before = request.generated_size()
            decoded = runtime.decode_one(request, item.sampler, item.executor)
            if decoded == 0:
                continue

            progress = True
            if request.generated_size() > before:
                stream_token(item.request_id, request.generated[-1])
```

这段 Python 是 C++ 控制流镜像。真实 ownership、错误和终态处理以 `ScheduledRequest`、`Runtime` 和单元测试为准。

| Python 条件 | C++ 对象 | 测试证据 |
| --- | --- | --- |
| `scheduled_requests` | `vector<ScheduledRequest>` | 多请求输入顺序固定 |
| `request.terminal()` | `Request::terminal()` | 完成、取消和失败请求停止推进 |
| `decode_one(...)` | `Runtime::decode_one()` | prefill/decode 与 Executor 隔离 |
| `decoded == 0` | 无进展分支 | 不产生占位 token 或 streaming event |
| `progress` | 外层退出条件 | 全部请求无进展时结束 |

## Admission 与请求资源

Admission 决定请求能否进入运行集合，Scheduler 决定已接纳请求的推进顺序。ServiceCore 在 Chapter09 管理 waiting/running 队列、容量、取消和超时；Chapter05 Scheduler 消费已经准备好的请求和预算。

`ScheduledRequest` 可以携带 `shared_ptr<RequestModelExecutor>`。真实模型只加载一次，每个请求拥有独立 Context；Scheduler 选中请求时调用对应 executor，terminal cleanup 后由 ServiceCore 释放请求资源。

| 阶段 | Runtime / Scheduler | Executor / Context |
| --- | --- | --- |
| Prefill | 校验 budget，记录 prompt history | 建立该请求各层 K/V |
| Decode | 采样、追加 token、检查终态 | 只推进所属 Context |
| 执行失败 | Request 进入 `Failed`，停止调度 | 返回错误，不生成占位 token |
| Cleanup | Request cache 清空，entry 移除 | Executor 引用释放，Context 销毁 |

Round-robin 提供可重复的 FIFO 基线。优先级、饥饿保护、chunked prefill 和 token/block budget 在 [`05.10 Continuous Batching`](./10_continuous_batching.md) 中扩展。

## 验证

```bash
bash code/chapter05_generation/scripts/run.sh
```

检查多个请求按固定顺序轮转；terminal 请求不再进入 `decode_one()`；两个 executor 的调用次数、token 和 Context 互不污染；executor 失败时 Request 进入 `Failed`，且不会对空 logits 采样。
