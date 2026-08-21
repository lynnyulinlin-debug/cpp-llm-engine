# 05.8 vLLM 对照：把最小 generation runtime 放到真实 serving 里看

到这里，你已经把 `forward` 变成了一条能逐 token 推进的最小 runtime。  
这一节不再重复讲生成流程，而是把本章这条最小链路放到 `vLLM` 里对照着看，帮你分清：

- 哪些是生成 runtime 的必需边界
- 哪些是高吞吐 serving 的扩展做法
- 哪些地方是 `vLLM` 真正做强的部分

## 本节概览

| 本章里的东西 | `vLLM` 里的对应关注点 | 为什么重要 | 读的时候重点看什么 |
| --- | --- | --- | --- |
| `prefill / decode` | continuous batching 下的请求推进 | 这是一条 runtime 的基本分段 | 一个请求什么时候进入 batch，什么时候离开 batch |
| `KV Cache` | `PagedAttention`、prefix caching、cache 管理 | cache 往往决定了生成是否真正高效 | cache 如何分块、复用、减少碎片 |
| `Scheduler` | 动态批处理与请求调度 | 吞吐和延迟要一起平衡 | 哪些请求先推进，哪些请求被合并进 batch |
| `sampler` | token 级输出与 streaming | 生成必须能持续吐 token | token 如何在调度下持续输出 |
| `Request state` | 多请求状态管理 | 多请求下必须先把上下文分开 | 每个请求的上下文怎样隔离、怎样复用 |

## 阅读入口

- 前置阅读：[`05.4 KV Cache`](./04_kv_cache.md)、[`05.5 Scheduler`](./05_scheduler.md)
- 源码直达：<a href="../_code/chapter05_generation/src/runtime.cpp"><code>runtime.cpp</code></a>、<a href="../_code/chapter05_generation/src/scheduler.cpp"><code>scheduler.cpp</code></a>、<a href="../_code/chapter05_generation/src/kv_cache.cpp"><code>kv_cache.cpp</code></a>
- 相关阅读：[`05.9 Request 生命周期`](./08_request_lifecycle.md)、[`05.10 Continuous Batching 入门`](./09_continuous_batching.md)、[`vLLM 生态对照`](../reference/ecosystem/vllm.md)

## 工程映射

本章已经把 generation 做成了一条可观察、可验证的状态流：

- `prefill` 先建状态
- `decode` 再逐 token 推进
- `KV Cache` 负责保留历史
- `scheduler` 负责推进顺序
- `sampler` 负责把 logits 变成下一步 token

`vLLM` 把这条最小链路扩展成面向高并发服务的 runtime。  
这里重点看的是，它把本章边界推到了哪里。

核心对应关系如下：

1. `prefill / decode` 对应请求的阶段推进
2. `KV Cache` 对应高吞吐 serving 里的状态与内存管理
3. `scheduler` 对应 batch 组织、请求推进和吞吐控制
4. `streaming` 对应 token 级结果持续输出

往下看 `vLLM` 时，先抓这几件事：

- `PagedAttention` 解决的是 cache 组织问题
- `continuous batching` 解决的是 batch 演化问题
- `prefix caching` 解决的是共享前缀复用问题
- `disaggregated prefill / decode` 解决的是更细的执行分层问题

这些都不是本章里必须一次做完的东西，但都建立在本章已经立住的那条主线上。  
把它们和本章对应起来，再看 `vLLM` 时就能看出它在解决什么问题。

如果从“为什么 `vLLM` 要长成这样”倒推，通常会得到几个约束：

- 单条请求不够，必须支持很多并发请求
- 只支持生成不够，必须控制吞吐和延迟
- cache 不能只存，必须会管理、会复用、会减少碎片
- scheduler 不能只轮转，必须能在 batch 演化里保持效率
- streaming 不能只是回调，必须能和调度、cache、状态管理一起工作

把这些约束摆出来后，就能看懂 `vLLM` 为什么和本章的最小 runtime 是同一条主线上的两个层次：

- 本章先把生成链路拆清楚
- `vLLM` 再把同一条链路做成高吞吐 serving runtime

## 学习边界

本节完成后，应当能够把下面几件事说清楚：

- 为什么 `prefill` 和 `decode` 不能混成一个模糊的循环
- 为什么 `KV Cache` 是生成 runtime 的核心状态
- 为什么 scheduler 在高吞吐 serving 里比“会跑”更重要
- 为什么 `vLLM` 的优化重点会落在 cache、batch 和请求推进上

把这些对应关系讲明白，就说明你已经能把本章的 runtime 组织方式放回真实 serving 语境里去看。

这一节不负责复刻 `vLLM` 的全部能力。

它只负责把本章最小 runtime 和 `vLLM` 的工程关注点对齐，区分本章已经出现的概念与真实系统中的扩展设计。

所以这里的边界是：

- 本章负责“能生成、能观察、能验证”
- `vLLM` 负责“高吞吐、高并发、低碎片”
- 本节负责把两者之间的对应关系讲清楚

参考页：

- [vLLM](../reference/ecosystem/vllm.md)
- <a href="../_code/chapter05_generation/src/runtime.cpp"><code>runtime.cpp</code></a>、<a href="../_code/chapter05_generation/src/scheduler.cpp"><code>scheduler.cpp</code></a>、<a href="../_code/chapter05_generation/src/kv_cache.cpp"><code>kv_cache.cpp</code></a>
