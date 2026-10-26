# 08.3 Memory and Bandwidth 内存与带宽瓶颈

推理引擎不只受算力限制，也经常受内存和带宽限制。性能分析需要判断问题是否来自数据移动、分配和 cache 组织。

**阅读入口**

- 前置阅读：[`02.1 Shared Underlying Storage`](../chapter02_tensor_ops/01_shared_ptr_memory.md)、[`05.4 KV Cache`](../chapter05_generation/04_kv_cache.md)
- 源码直达：<a href="../_code/chapter05_generation/src/kv_cache.cpp"><code>kv_cache.cpp</code></a>、<a href="../_code/chapter06_quantization/src/quantized_linear.cpp"><code>quantized_linear.cpp</code></a>、<a href="../_code/chapter06_quantization/src/dequantize_op.cpp"><code>dequantize_op.cpp</code></a>
- 相关阅读：[`05.11 Block KV Cache 阅读入口`](../chapter05_generation/10_block_kv_cache.md)、[`06.8 量化在 Serving Runtime 里的位置`](../chapter06_quantization/08_quantization_in_serving_runtime.md)

## 常见信号

- 计算量不大，但 latency 很高
- batch 变大后吞吐没有明显改善
- 长上下文下显存快速上涨
- 每步 decode 都出现重复分配或拷贝
- 量化后显存下降，但端到端速度没有明显改善

这些信号说明问题可能不在算子本身，而在数据移动和资源复用。

## 先拆四类内存成本

| 成本 | 看什么 |
| --- | --- |
| weight | 权重 dtype、加载方式、是否重复持有 |
| activation | 中间 tensor 生命周期和 workspace |
| KV cache | cache shape、block、增长和释放 |
| transfer | host / device 拷贝、同步和 staging buffer |

后续优化不要只问“能不能少算”，也要问“能不能少搬、少分配、少等待”。

## 两个常见误判

第一，量化后显存下降，不代表端到端一定更快。如果当前实现每次都把 INT8 权重反量化成浮点再计算，收益可能主要体现在加载和存储，不一定体现在 decode latency。

第二，KV cache 增长不是单纯的容量问题。长上下文下真正影响性能的，可能是 cache layout、block 管理、访存连续性和调度时的 cache 复用策略。

## 用当前基线先排除什么

当前 `chapter07_benchmark_demo` 的真实样例里，`prefill` 占 `50%`，`decode` 占 `25%`，`sampling` 和 `output` 各占 `8.33%`。这组比例至少说明一件事：在这个最小 CPU baseline 里，第一步还不应该优先怀疑输出链路。

更合适的排查顺序是：

1. 先确认 `prefill` 路径里有没有不必要的数据拷贝或重复分配
2. 再确认 `decode` 路径是否随着输出长度增长而放大
3. 最后再看 `sampling`、`output` 是否只是当前样例较轻，还是本身就不是主成本

这就是内存与带宽页要做的事：先排除“看起来像瓶颈、但当前数据没有证明”的方向。
