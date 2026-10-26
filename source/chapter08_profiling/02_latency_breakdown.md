# 08.2 Latency Breakdown 延迟拆解

端到端 latency 是结果，不是原因。把一次生成请求拆成几个阶段，再根据各阶段耗时定位慢点。

**阅读入口**

- 前置阅读：[`08.1 Profiling Basics`](./01_profiling_basics.md)、[`05.9 Request 生命周期`](../chapter05_generation/08_request_lifecycle.md)
- 源码直达：<a href="../_code/chapter05_generation/src/runtime.cpp"><code>runtime.cpp</code></a>、<a href="../_code/chapter05_generation/src/scheduler.cpp"><code>scheduler.cpp</code></a>、<a href="../_code/chapter07_benchmark/src/benchmark.cpp"><code>benchmark.cpp</code></a>
- 相关阅读：[`05.10 Continuous Batching 入门`](../chapter05_generation/09_continuous_batching.md)、[`07.3 Benchmark Metrics`](../chapter07_benchmark/03_benchmark_metrics.md)

## 最小阶段拆分

| 阶段 | 主要工作 | 常见瓶颈 |
| --- | --- | --- |
| input | token、position、请求参数准备 | 输入处理和拷贝 |
| prefill | 处理 prompt，写入初始 KV cache | attention 和大矩阵计算 |
| decode | 每步生成一个或少量 token | kernel launch、cache 读取、调度 |
| sampling | logits 到 token | top-k / top-p、同步和拷贝 |
| output | 返回结果或流式输出 | 回调、序列化、网络 |

## 先问三个问题

1. 首 token 慢，还是后续 token 慢？
2. 单请求慢，还是并发后变慢？
3. 长 prompt 慢，还是长输出慢？

这三个问题能把问题大致分到 prefill、decode、调度或输出层。不要一开始就盯着某个函数名，先确认慢的是哪一段链路。

## 最小拆解表

一次 benchmark 至少应该能落到下面这种表里：

| 阶段 | 时间 | 占比 | 判断 |
| --- | --- | --- | --- |
| input | 2ms | 3% | 不是主瓶颈 |
| prefill | 38ms | 55% | 长 prompt 下重点看 attention / matmul |
| decode | 24ms/token | - | 短输出看 launch，长输出看 cache |
| sampling | 4ms/token | - | 如果 top-k / top-p 复杂，要单独看 |
| output | 1ms/token | - | 流式服务里可能被网络放大 |

这张表不要求一开始完全精确，但必须把首 token、后续 token 和输出阶段分开。否则“端到端 latency 变慢”无法变成可行动的优化判断。

## 用当前教程基线怎么拆

对当前 `chapter07_benchmark_demo` 的真实输出，可以先做最小拆解：

| 阶段 | tokens | 时间 | 占比 | 第一判断 |
| --- | --- | --- | --- |
| input | 5 | 0.05ms | 8.33% | 当前样例里不是主瓶颈 |
| prefill | 5 | 0.30ms | 50.00% | 当前样例里占主要时间，先看 prompt 处理路径 |
| decode | 2 | 0.15ms | 25.00% | 当前样例较短，但已经能单独观察每步生成成本 |
| sampling | 2 | 0.05ms | 8.33% | 当前样例里较轻，先不作为第一优化点 |
| output | 2 | 0.05ms | 8.33% | 当前样例里较轻，服务化后可能被网络放大 |
| benchmark total | 7 | 0.60ms | 100% | 先作为 baseline，后续所有实验都对照它 |

这个拆解还很粗，但已经足够支持两件事：

1. 把端到端结果拆回阶段，不再只看总时间
2. 为下一步 profiling 指定入口，先看 `prefill` 路径，再决定是否继续细拆 `decode`

如果后续把样例扩展到更长输出，`decode latency/token` 才会变成更稳定的观察指标。
