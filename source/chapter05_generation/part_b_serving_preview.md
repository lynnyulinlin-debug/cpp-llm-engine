# 05-B Serving Preview 服务化预览

这一部分观察多个请求如何进入、退出并共享 Generation Runtime 的执行资源。

## 目标

下面的对象建立服务化概念和最小验证，Chapter09 再把它们接入持续服务：

| 能力 | 本章处理 | Chapter09 接管 |
| --- | --- | --- |
| Admission | context budget、终态请求拒绝 | 限流、队列和优先级 |
| Continuous Batching | `StepBatchPlanner` 按 token budget 混合 prefill chunk/decode，并按最少进度轮换；`ContinuousBatchScheduler` 推进一个 step | 抢占、业务优先级和并行 batch |
| Block KV Cache | `KvBlockPool` 分配物理 block ID，`BlockTable` 映射 token 到 block/slot | 分层 K/V payload 和 PagedAttention |
| Streaming | callback 和 finish reason 的基本语义 | HTTP / OpenAI-compatible response |

## 进阶阅读顺序

完成 Generation Core 后，准备进入 Chapter09 或阅读 vLLM scheduler 时，按以下顺序学习。

1. [`Scheduler`](./05_scheduler.md) 中的 admission 和多请求推进
2. [`Request 生命周期`](./09_request_lifecycle.md)
3. [`Continuous Batching`](./10_continuous_batching.md)
4. [`Block KV Cache`](./11_block_kv_cache.md)
5. [`vLLM / Mini-SGLang / SGLang 对照`](./08_vllm_comparison.md)

## 验证入口

```bash
bash code/chapter05_generation/scripts/run.sh
```

重点检查：取消和超时请求不会再次调度，decode 优先使用预算，长 prompt 按 chunk 轮换推进且只有 final chunk 采样；block 容量不足或 executor 失败时不遗留本 step 预留，终态释放后物理 block ID 可以复用。

## Chapter09 如何接入

Chapter05-B 留下 `Request`、`Runtime`、`ContinuousBatchScheduler`、`KvBlockPool`、`KVCache` 和 streaming callback。Chapter09 用 `dispatch_step()` 将 waiting/running 队列接到 token/block budget，并在 terminal release 时统一归还 block。
