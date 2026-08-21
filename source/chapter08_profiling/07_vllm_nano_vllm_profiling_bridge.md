# 08.7 vLLM / nano-vLLM Profiling 阅读桥

这一页把本章方法放回 vLLM / nano-vLLM。读真实项目的优化讨论时，不要只记住某个技巧，先判断它在解决哪一类瓶颈。

## 阅读入口

- 前置阅读：[`08.1 Profiling Basics`](./01_profiling_basics.md)、[`08.6 Optimization Loop`](./06_optimization_loop.md)
- 源码直达：<a href="../_code/chapter05_generation/src/scheduler.cpp"><code>scheduler.cpp</code></a>、<a href="../_code/chapter05_generation/src/kv_cache.cpp"><code>kv_cache.cpp</code></a>、<a href="../_code/chapter07_benchmark/src/benchmark.cpp"><code>benchmark.cpp</code></a>
- 相关阅读：[`05.8 vLLM 对照`](../chapter05_generation/07_vllm_comparison.md)、[`07.7 主流对照`](../chapter07_benchmark/07_framework_comparison.md)、[`vLLM 生态对照`](../reference/ecosystem/vllm.md)

## 对照阅读坐标

| 真实项目里的主题 | 本教程对应基础 | 先问的问题 |
| --- | --- | --- |
| continuous batching | `chapter05` scheduler | 改善的是吞吐、等待时间，还是 GPU 利用率 |
| PagedAttention / block cache | `chapter05` KV cache | 改善的是显存碎片、长上下文，还是 cache 寻址 |
| quantized kernel | `chapter06` 量化路径 | 省的是权重带宽、显存，还是计算时间 |
| benchmark report | `chapter07` 结果解读 | 条件是否一致，baseline 是谁 |
| profiling trace | `chapter08` 性能归因 | 热点在哪个阶段，证据是否支持改动 |

## 读优化 PR 或文档时怎么判断

先找它的目标指标：prefill latency、decode latency、throughput、显存，还是尾延迟。

再找它改变的层：scheduler、cache、kernel、memory pool、输入链路，还是服务输出。

最后找它的证据：有没有固定条件下的 A/B 对比，有没有正确性验证，有没有说明适用边界。

这样读，开源项目就不是“技巧清单”，而是一组可以映射回本教程主线的工程决策。

## 学完本章后怎么回看开源项目

读 vLLM / nano-vLLM 的性能相关代码时，可以按这条顺序走：

1. 先看 benchmark 或 issue 里报告的目标指标
2. 再确认瓶颈属于 prefill、decode、cache、scheduler 还是 kernel
3. 再看改动实际落在哪个模块
4. 最后看 A/B 数据和正确性验证是否支持这个改动

本教程的作用是提供这套判断框架。真实项目会更复杂，但大多数优化仍然可以拆回“现象、证据、改动、复测、决策”这条链路。

## 从教程 trace 过渡到真实项目

当前教程里的最小 trace 已经能把 `run_prefill_attention`、`decode_step`、`sample_logits`、`flush_stream` 这些事件拆开。读 vLLM / nano-vLLM 时，可以把它们替换成更真实的工程对象：

| 教程里的最小事件 | 真实项目里可能对应什么 | 阅读时先问什么 |
| --- | --- | --- |
| `run_prefill_attention` | prefill attention backend / kernel | prefill 主要在算，还是在搬数据 |
| `write_kv_cache` | block cache write / page update | cache layout 是否影响访存 |
| `decode_step` | decode scheduler + model executor | 慢在调度、kernel，还是同步 |
| `sample_logits` | logits processor / sampler | sampling 是否已经变成显著成本 |
| `flush_stream` | streaming response / network flush | 这是模型慢，还是服务输出慢 |

这样过渡的意义是：教程先教你识别“证据长什么样”，再去开源项目里识别更复杂版本，而不是一开始就陷进实现细节。

## 从 hotspot 表过渡到真实 profiler

当前教程里已经有一张最小 hotspot 表，里面会把 `run_prefill_attention`、`decode_step`、`sample_logits` 这些符号按耗时列出来。把这张表映射到真实项目时，可以这样理解：

| 教程里的 hotspot | 真实项目里常见替身 | 先判断什么 |
| --- | --- | --- |
| `run_prefill_attention` | attention backend kernel / fused attention path | prefill 是否就是当前 top hotspot |
| `decode_step` | model executor / decode loop / scheduler path | decode 主循环是不是吞吐上限 |
| `sample_logits` | sampler / logits processor | 采样逻辑是否被复杂策略放大 |
| `flush_stream` | streaming response / transport flush | 尾延迟是在模型内，还是在服务外层 |

这样读 vLLM / nano-vLLM 时，就不是“看到一个火焰图就开始猜”，而是先把最重的几个符号放回 `prefill / decode / output` 这条主线里。

## 从教程 JSON 过渡到真实工具

当前教程基线已经能输出一份最小 JSON，里面同时包含 `phases`、`trace_events` 和 `hotspots`。这和真实项目里常见的 profiler 导出物很接近，只是规模更小、字段更少。

学习者需要建立的不是某个工具按钮记忆，而是这个判断顺序：

1. 先看阶段层，确认慢在哪一段
2. 再看事件层，确认这一段里发生了什么
3. 最后看 hotspot 层，决定先优化哪几个符号

真实工具会更复杂，但这三层关系不变。

如果工具再往前一步提供时间轴视图，例如 Chrome trace 或 Nsight Systems，读法也不变，只是多了一层“事件是串行、重叠，还是被同步点打断”的观察角度。

在当前教程的简化时间轴里，这层信息已经通过两种最小字段出现：

- `tid`：区分 `host-thread` 和 `compute-stream`
- `sync_decode_result`：提示 host 侧等待 decode 结果的同步点

这就够让学习者建立一个真实工程判断：慢，不一定只是算得慢，也可能是线程之间在等。

## 多请求时间轴该怎么看

当前教程基线现在还有一个最小多请求时间轴样例。它不尝试复刻 continuous batching 全貌，只回答一个更实际的问题：当 `req-A` 还在跑时，`req-B` 进来后，你在时间轴里应该看什么。

最小样例里只保留三类信号：

- `req-A prefill / decode`
- `req-B prefill / decode`
- 一个 `sync_batch_merge` 等待点

读法也很简单：

1. 先看两个请求有没有重叠，确认系统是不是已经在复用 stream 或执行窗口
2. 再看等待点是不是出现在 batch 合并或结果回收附近
3. 最后再问吞吐改善来自重叠执行，还是只是把等待后移了

这就是把单请求 profiling 过渡到 serving 场景的最小一步。
