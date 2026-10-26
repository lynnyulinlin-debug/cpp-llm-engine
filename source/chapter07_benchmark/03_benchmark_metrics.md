# 07.3 Benchmark Metrics Benchmark 指标


## 固定性能指标

定义统一的性能指标，衡量模型的 prefill 延迟、逐 token 延迟和吞吐能力。  
没有统一指标，就没法公平比较不同实现。

读完后应能做到：

- 理解 latency、prefill 和 throughput 的区别
- 理解为什么不同阶段需要不同指标
- 理解指标如何反映系统性能

**阅读入口**

- 前置阅读：[`07.2 End-to-End Correctness Testing`](./02_e2e_correctness.md)、[`chapter06_quantization`](../chapter06_quantization/intro.md)
- 源码直达：<a href="../_code/chapter07_benchmark/src/benchmark.cpp"><code>benchmark.cpp</code></a>、<a href="../_code/chapter07_benchmark/src/test_helpers.cpp"><code>test_helpers.cpp</code></a>、<a href="../_code/chapter07_benchmark/examples/main_benchmark.cpp"><code>main_benchmark.cpp</code></a>
- 相关阅读：[`07.4 Benchmark Script`](./04_benchmark_script.md)、[`推理性能分析`](../reference/profiling.md)

## 指标设计

没有统一指标，就无法比较不同实现、不同硬件或不同优化策略。  
benchmark 不是“跑个数字”，而是把系统行为变成可比较的证据。

核心指标：

- Prefill latency
- Token latency
- Throughput

这三个指标分别对应不同的真实场景：

- `Prefill latency` 看首轮输入处理
- `Token latency` 看逐步生成的响应速度
- `Throughput` 看系统整体吞吐能力

把它们混成一个数，工程意义会大幅下降。  
更合理的做法是让每个指标回答不同问题，而不是让一个指标解释全部现象。

从用户视角看，prefill 影响首次响应体验，token latency 影响连续输出的流畅度，throughput 决定服务能扛住多少并发请求。

最小指标记录结构可以写成：

```cpp
struct BenchmarkResult {
    double prefill_ms;
    double token_ms;
    double throughput_tok_s;
};
```

最小测量条件清单应包括：硬件型号（CPU / GPU）、batch size、sequence length、量化配置（FP16 / INT8）、CUDA 版本，以及采样策略（greedy / temperature）。这些条件应当和 `BenchmarkResult` 一起记录，否则指标就无法在不同环境之间复现或比较。

实现建议：

- 先固定测量口径
- 再记录不同配置下的结果
- 指标要和实际运行场景对应

## 验证与排查

验证时重点看：

- 指标定义清晰
- 同一场景下可重复测量
- 不同实现可公平比较

如果指标定义清晰，读者就能一眼看出优化到底是在改善首 token、持续 decode，还是整体吞吐。

常见问题：

- 混淆 prefill 和 decode
- 只看平均值，不看波动
- 不记录测量条件
