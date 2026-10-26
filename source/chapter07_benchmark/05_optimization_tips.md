# 07.5 Optimization Tips 优化建议


## 先测量再选择优化

列出常见的可选优化方向，例如 CUDA Graph、memory pool 和 async copy，帮助后续定位性能提升空间。

读完后应能做到：

- 理解这些优化项各自适合解决什么问题
- 理解为什么优化建议要放在 benchmark 之后
- 理解性能优化不等于先优化所有地方

**阅读入口**

- 前置阅读：[`07.3 Benchmark Metrics`](./03_benchmark_metrics.md)、[`07.4 Benchmark Script`](./04_benchmark_script.md)
- 源码直达：<a href="../_code/chapter07_benchmark/src/benchmark.cpp"><code>benchmark.cpp</code></a>、<a href="../_code/chapter07_benchmark/examples/main_benchmark.cpp"><code>main_benchmark.cpp</code></a>
- 相关阅读：[`08.4 Kernel Launch and CUDA Graph`](../chapter08_profiling/04_kernel_launch_and_cuda_graph.md)、[`foundation/07 CUDA 常用库`](../foundation/07_cuda_common_libraries.md)

## 优化边界

优化必须建立在测量结果之上。先知道瓶颈在哪里，再决定是否引入优化手段。

可选优化方向：

- CUDA Graph
- memory pool
- async copy

这些优化项都能在某些工作负载下减少重复开销，但并不一定对所有场景都有效。教程把它们放在 benchmark 之后，就是为了避免“先上优化，再找问题”。

优化建议适合写成“优化备忘录”：列出常见方向、各自适合解决的瓶颈，以及为什么必须先测量后决定。

可以把优化建议理解成一个决策表：当瓶颈在调度时看 scheduler，当瓶颈在 kernel launch 时看 CUDA Graph，当瓶颈在内存分配时看 memory pool，当瓶颈在拷贝时看 async copy。

在工程实践中，建议的优化优先级通常是：先看内存分配和释放是否为热点，memory pool 往往收益明确且实现相对简单；再看 kernel launch 开销，CUDA Graph 的收益取决于 batch size 和模型规模；最后看计算与拷贝重叠，async copy 的收益取决于 kernel 时间和拷贝时间的比值。不是每个项目都需要把这三类优化全部做完。

这一章只要求读者先理解最小可用的 memory pool 思路。

- 复用临时 buffer，而不是每次都重新分配
- buffer 用完后放回池中，不要立刻释放给系统
- 重点观察分配次数、峰值内存和运行抖动
- 完整的 allocator 设计留到后续进阶

这样可以先把主线聚焦在可测量的性能提升上，再去讨论通用 allocator 设计。

最小的 memory pool 可以这样理解：

- 预先准备一组固定大小或按 size class 分组的 buffer
- 申请时先从池里拿，只有池里没有合适对象时才新分配
- 归还时把 buffer 放回池中，而不是立刻释放给系统
- benchmark 里重点观察分配次数、峰值内存和运行抖动

这类实现不需要一开始就做成完整 allocator。教程要让读者先看到“复用”为什么能降低开销，而不是先被复杂内存管理细节淹没。

和 CUDA 工具箱的关系可以再明确一点：

- `CUDA Graph` 更偏执行调度和 launch 开销控制
- `memory pool` 更偏分配器和复用策略
- `async copy` 更偏数据搬运和计算重叠

它们不一定是独立“库名”，但都属于 CUDA 侧常见的性能手段。教程这里先把概念和使用场景讲清楚，后面如果真的要实现，再去对照具体 API。

要把这些手段和具体 CUDA 工具对应起来，可以回看
[`foundation/07_cuda_common_libraries`](../foundation/07_cuda_common_libraries.md)。

使用建议：

- 先测量，再优化
- 先确认热点，再引入复杂机制
- 每个优化项都要对应一个可见收益

## 验证与排查

验证时重点看：

- 优化前后有明确对比
- 优化不破坏正确性
- 性能收益可以被 benchmark 观察到

如果优化能明确改善 benchmark 指标，同时不引入新的正确性问题，这个优化才值得保留。

常见问题：

- 盲目引入优化，增加复杂度
- 优化项和实际瓶颈不匹配
- 只看峰值，不看稳定性
