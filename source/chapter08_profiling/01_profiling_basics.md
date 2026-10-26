# 08.1 Profiling Basics Profiling 基础

benchmark 回答“快了多少”，profiling 回答“为什么这样”。先把这两件事拆开，避免看到一个性能数字后直接猜瓶颈。

**阅读入口**

- 前置阅读：[`chapter08 导读`](./intro.md)、[`07.3 Benchmark Metrics`](../chapter07_benchmark/03_benchmark_metrics.md)
- 源码直达：<a href="../_code/chapter07_benchmark/src/benchmark.cpp"><code>benchmark.cpp</code></a>、<a href="../_code/chapter07_benchmark/src/test_helpers.cpp"><code>test_helpers.cpp</code></a>
- 相关阅读：[`推理引擎 Profiling`](../reference/profiling.md)、[`08.2 Latency Breakdown`](./02_latency_breakdown.md)

## 先看什么

| 问题 | benchmark | profiling |
| --- | --- | --- |
| 回答什么 | 快多少、慢多少 | 为什么快、为什么慢 |
| 输出什么 | latency、throughput、显存、误差 | 热点、阶段耗时、调用路径、等待点 |
| 用来做什么 | 决定是否值得继续 | 决定下一步改哪里 |

最小原则是：没有 benchmark，不要谈优化收益；没有 profiling，不要轻易判断瓶颈原因。

### 推理引擎里先看哪些层

- 计算：MatMul、attention、dequantize、sampling
- 内存：分配、拷贝、KV cache、workspace
- 调度：request admission、batch 组装、decode step
- 流程：prefill / decode 切换、同步点、输出回调

先把热点归到正确层级，再决定是否需要更细的工具。

## 记录结果

第一次做 profiling 时，不要直接写“attention 慢”或“cache 慢”。先写成可复查记录：

| 字段 | 示例 |
| --- | --- |
| 场景 | batch=1, prompt=512, output=128, FP16 |
| benchmark 现象 | decode latency 从 18ms/token 变成 24ms/token |
| profiling 证据 | decode step 中重复出现小 kernel 和同步点 |
| 初步归因 | kernel launch / host-device sync，而不是 matmul 计算量 |
| 下一步实验 | 固定输入，只改 decode step 的同步或复用策略 |

这样写的好处是，后面即使结论错了，也能知道错在 benchmark、profiling 解释，还是实验设计。

### 一个真实基线样例

下面这组输出来自当前仓库里已经可运行的最小 benchmark demo：

```bash
cmake -S code/chapter07_benchmark -B /tmp/chapter07_benchmark_build
cmake --build /tmp/chapter07_benchmark_build
/tmp/chapter07_benchmark_build/chapter07_benchmark_demo
```

```text
[chapter07] benchmark demo
unit checks: true
e2e checks: true
result name: demo
result items/s: 4000
samples: 6
chapter07 benchmark summary
input: tokens=5, elapsed_ms=0.05, share=8.33%, tokens_per_second=100000.00
prefill: tokens=5, elapsed_ms=0.30, share=50.00%, tokens_per_second=16666.67
decode: tokens=2, elapsed_ms=0.15, share=25.00%, tokens_per_second=13333.33
sampling: tokens=2, elapsed_ms=0.05, share=8.33%, tokens_per_second=40000.00
output: tokens=2, elapsed_ms=0.05, share=8.33%, tokens_per_second=40000.00
benchmark: tokens=7, elapsed_ms=0.60
```

这组输出能先回答三件事：

- 正确性检查通过没有
- baseline 吞吐大致是多少
- 当前流程已经拆成了 `input / prefill / decode / sampling / output`

但它还不能回答热点在哪个函数、同步点在哪里、有没有 cache miss。这正是 profiling 要补上的部分。

### 再往下一层看什么

如果阶段汇总显示 `prefill` 最重，下一步不要直接改代码，先补两类证据：

- `trace events`：看 `prefill` 内部到底是哪几个事件在耗时
- `hotspot summary`：看排序最靠前的函数或符号是谁

在当前教程基线里，这两层证据已经能落成真实输出。它们的关系是：

- 阶段汇总告诉你“慢在哪一段”
- trace 事件告诉你“这段里发生了什么”
- hotspot 排序告诉你“先看哪几个点最值当”
