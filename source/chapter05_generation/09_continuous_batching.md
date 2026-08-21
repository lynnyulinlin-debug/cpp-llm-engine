# 05.10 Continuous Batching 入门

朴素生成通常按“一个请求跑完再跑下一个”理解，但真实 serving runtime 更关心吞吐和等待时间。Continuous Batching 的核心是：每个 decode step 都重新组织当前可运行的请求，让新请求可以插入，让已完成请求及时离开。

## 阅读入口

- 前置阅读：[`05.5 Scheduler`](./05_scheduler.md)、[`05.9 Request 生命周期`](./08_request_lifecycle.md)
- 源码直达：<a href="../_code/chapter05_generation/src/scheduler.cpp"><code>scheduler.cpp</code></a>、<a href="../_code/chapter05_generation/src/runtime.cpp"><code>runtime.cpp</code></a>
- 相关阅读：[`05.11 Block KV Cache 阅读入口`](./10_block_kv_cache.md)、[`05.8 vLLM 对照`](./07_vllm_comparison.md)、[`vLLM 生态对照`](../reference/ecosystem/vllm.md)

## 它解决什么问题

如果 batch 在请求开始时固定，短请求会被长请求拖住，新请求也要等整批结束后才能进入。Continuous Batching 把 batch 的边界从“请求集合”改成“每个 step 的可执行集合”。

这样 runtime 每一步都可以做三件事：

- 把新请求放进 prefill
- 让正在 decode 的请求继续生成一个 token
- 把结束、取消或资源不足的请求移出运行集合

## 最小调度循环

一个便于入门的循环可以写成下面这个形状：

```text
while has_work:
  admit_waiting_requests()
  run_prefill_for_new_requests()
  run_one_decode_step_for_active_requests()
  sample_next_tokens()
  update_request_status()
  release_finished_requests()
```

真实 vLLM 会把这件事做得复杂得多：它需要考虑 token budget、block budget、prefill 和 decode 的混排、抢占、streaming output，以及 GPU batch 形状。但入门时先抓住一个事实：scheduler 调度的不是完整请求的一次执行，而是每个 step 里哪些 token 应该被推进。

## 和本教程最小实现的关系

本教程当前的 scheduler 先让请求推进规则变得可见，不追求完整 serving 性能。这个小节补上的，是从最小 scheduler 走向 vLLM / nano-vLLM 的关键差距：真实 runtime 会持续重组 batch，而不是把 batch 当成一次性输入。

后续如果要继续实现，优先补三件事：

- waiting / running / finished 队列分离
- 每个 step 显式计算 token budget
- decode 结束后立即更新请求和 cache 状态
