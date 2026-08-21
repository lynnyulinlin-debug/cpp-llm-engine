# 08.0 Explain Why It Is Slow 性能归因与优化闭环

这一章先回答一个优化问题：benchmark 已经告诉我们快慢变化之后，怎样解释原因，并把优化变成可复查的工程闭环？

`chapter07` 负责证明系统“对不对、快多少”；本章负责继续追问“为什么快、为什么慢、下一步该改哪里”。它不新增一套独立 runtime，而是在已有 benchmark 结果上建立 profiling、A/B 实验和优化决策流程。

## 本章和 vLLM / nano-vLLM 的联系

读 vLLM / nano-vLLM 时，很多性能结论都来自 profiling 和 A/B 对比：瓶颈可能在 prefill、decode、KV cache、scheduler、kernel launch、内存分配或数据搬运。本章先建立归因方法，把真实项目里的性能优化拆回可验证的层级。

## 本章概览

| 先看什么 | 这一节讲什么 | 为什么重要 | 对应小节 |
| --- | --- | --- | --- |
| profiling 和 benchmark 怎么分工 | 结果对照与原因解释 | 先避免把数字误当成原因 | [`08.1 Profiling Basics`](./01_profiling_basics.md) |
| 延迟应该怎么拆 | prefill、decode、sampling、output | 先定位慢在哪个阶段 | [`08.2 Latency Breakdown`](./02_latency_breakdown.md) |
| 内存和带宽怎么看 | 分配、拷贝、KV cache、带宽瓶颈 | 先判断是不是数据移动问题 | [`08.3 Memory and Bandwidth`](./03_memory_bandwidth.md) |
| kernel launch 怎么影响性能 | CUDA Graph、launch overhead、同步点 | 先理解小 batch / decode 的固定开销 | [`08.4 Kernel Launch and CUDA Graph`](./04_kernel_launch_and_cuda_graph.md) |
| A/B 实验怎么设计 | 固定变量、单点改动、回滚标准 | 先让优化结论可复查 | [`08.5 A/B Experiment`](./05_ab_experiment.md) |
| 优化如何闭环 | hypothesis、change、measure、decide | 先把优化变成流程而不是猜测 | [`08.6 Optimization Loop`](./06_optimization_loop.md) |
| 怎么读真实项目优化 | vLLM / nano-vLLM profiling 坐标 | 先把本章方法映射到开源项目 | [`08.7 vLLM / nano-vLLM Profiling 阅读桥`](./07_vllm_nano_vllm_profiling_bridge.md) |
| 本章怎么收口 | profiling 和优化边界 | 让后续改动回到证据链 | [`08.8 本章小结`](./summary.md) |

## 阅读入口

- 前置阅读：[`chapter07_benchmark`](../chapter07_benchmark/intro.md)、[`07.6 Result Interpretation`](../chapter07_benchmark/06_result_interpretation.md)
- 源码直达：<a href="../_code/chapter07_benchmark/src/benchmark.cpp"><code>benchmark.cpp</code></a>、<a href="../_code/chapter07_benchmark/src/test_helpers.cpp"><code>test_helpers.cpp</code></a>、<a href="../_code/chapter05_generation/src/runtime.cpp"><code>runtime.cpp</code></a>
- 相关阅读：[`推理引擎 Profiling`](../reference/profiling.md)、[`推理引擎系统补强图`](../reference/inference_engine_systems.md)、[`回归测试`](../reference/regression_testing.md)

## 这章在解决什么

这一章只做一件事：把“性能数字”变成“可行动的优化判断”。当 benchmark 显示 token latency 变差时，归因范围应先覆盖 kernel、内存、调度、cache、输入和运行条件。

## 这章怎么做

本章使用一个固定流程：

1. 先确认 benchmark 条件一致
2. 再拆 prefill / decode / sampling / output
3. 再判断热点属于计算、内存、调度还是流程开销
4. 再设计只改变一个变量的 A/B 实验
5. 最后按正确性、性能收益和复杂度决定保留或回滚

本章的产物不是一个新 runtime，而是一份可以复查的优化记录。最小记录应包含：

- benchmark 条件：硬件、模型、输入长度、输出长度、batch、dtype
- profiling 结论：热点阶段、主要等待点、证据来源
- A/B 对照：baseline、experiment、指标变化、正确性结果
- 工程决策：保留、回滚，还是继续拆分下一轮实验

## 收口与下一步

到本章结束，应能形成一份完整的优化记录：改了什么、为什么改、改善了哪个指标、有没有破坏正确性、是否值得保留。

这章是主线从“能实现”走向“能优化”的入口。后续如果继续扩展代码，优先把 profiling 记录、A/B 报告和 benchmark 结果放到同一套可复查格式里。
