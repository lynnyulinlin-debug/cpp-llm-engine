# 08.4 Kernel Launch and CUDA Graph Kernel 启动开销

decode 阶段经常是小步循环：每一步只生成少量 token，但会反复触发多个 kernel。此时固定开销可能比单个 kernel 的计算时间更显眼。

## 阅读入口

- 前置阅读：[`03.4 cuBLAS Implementation`](../chapter03_matmul/04_cublas_implement.md)、[`07.5 Optimization Tips`](../chapter07_benchmark/05_optimization_tips.md)
- 源码直达：<a href="../_code/chapter03_matmul/src/matmul_op.cpp"><code>matmul_op.cpp</code></a>、<a href="../_code/chapter07_benchmark/src/benchmark.cpp"><code>benchmark.cpp</code></a>
- 相关阅读：[`foundation/07 CUDA 常用库`](../foundation/07_cuda_common_libraries.md)、[`08.5 A/B Experiment`](./05_ab_experiment.md)

## 什么时候考虑 CUDA Graph

适合考虑：

- decode step 形状稳定
- kernel launch 开销在 profiling 里很明显
- 同一段计算图会重复执行很多次

不适合一开始就做：

- shape 经常变化
- runtime 边界还不稳定
- 正确性和 benchmark 还没有固定

CUDA Graph 是优化手段，不是结构替代。先用 profiling 证明 launch overhead 是问题，再决定是否引入它。

## 读真实项目时看什么

vLLM / nano-vLLM 这类项目会把 decode 路径、attention backend 和 kernel 调用组织得更复杂。读的时候先看：

- 哪些路径是 prefill 专用
- 哪些路径是 decode 专用
- 哪些形状可以被缓存或复用
- kernel 选择是否依赖 batch、dtype、cache layout

这样看，CUDA Graph 和 fused kernel 就不会变成孤立技巧，而是 runtime 稳定之后的优化选择。

## 最小判断边界

如果 profiling 显示大部分时间花在大矩阵计算里，优先看 GEMM / attention kernel；如果显示每个 decode step 都有大量很短的 kernel 和同步点，再考虑 CUDA Graph、kernel fusion 或减少 host-device 往返。

换句话说，CUDA Graph 适合解决“重复提交稳定计算图”的固定开销，不适合替代尚未稳定的 runtime 结构。

## 一个最小事件级样例

当前教程代码已经能输出一组最小 trace 事件：

```text
chapter07 trace events
input | prepare_tokens | start_ms=0.00 | duration_ms=0.03 | note=token ids and positions
input | copy_request_state | start_ms=0.03 | duration_ms=0.02 | note=request metadata
prefill | embed_prompt | start_ms=0.05 | duration_ms=0.08 | note=prompt embedding
prefill | run_prefill_attention | start_ms=0.13 | duration_ms=0.14 | note=largest prefill slice
prefill | write_kv_cache | start_ms=0.27 | duration_ms=0.03 | note=initial cache materialization
decode | decode_step | start_ms=0.30 | duration_ms=0.09 | note=single token decode loop
decode | sample_logits | start_ms=0.39 | duration_ms=0.06 | note=logits to next token
sampling | topk_filter | start_ms=0.45 | duration_ms=0.03 | note=sampling filter
sampling | commit_token | start_ms=0.48 | duration_ms=0.02 | note=append generated token
output | format_response | start_ms=0.50 | duration_ms=0.03 | note=prepare response payload
output | flush_stream | start_ms=0.53 | duration_ms=0.05 | note=emit partial output
output | finalize_request | start_ms=0.58 | duration_ms=0.02 | note=request teardown
```

这组样例还不是 GPU kernel trace，但它已经提供了比阶段汇总更细的一层证据：

- `prefill` 里最大的单点事件是 `run_prefill_attention`
- `decode` 里已经能把 `decode_step` 和 `sample_logits` 分开
- `output` 里能看到 `flush_stream` 比 `format_response` 更重

这正是后面读真实 CUDA trace 时的最小心智模型。先知道自己在找哪一类事件，再去判断是否值得引入 CUDA Graph 或 fused kernel。

## 再加一层 hotspot 排序

当前教程代码还会输出一份最小 hotspot 汇总：

```text
chapter07 hotspot summary
prefill | run_prefill_attention | self_ms=0.14 | total_ms=0.17 | calls=1 | reason=largest prefill compute slice
decode | decode_step | self_ms=0.09 | total_ms=0.15 | calls=1 | reason=single token decode loop
prefill | embed_prompt | self_ms=0.08 | total_ms=0.08 | calls=1 | reason=prompt embedding path
decode | sample_logits | self_ms=0.06 | total_ms=0.06 | calls=1 | reason=decode tail after model step
output | flush_stream | self_ms=0.05 | total_ms=0.05 | calls=1 | reason=response emission dominates output
```

这张表的价值在于，它把“该先看哪个事件”直接排了序：

- 先看 `run_prefill_attention`，因为它是当前最重的单点
- 再看 `decode_step`，因为它决定 decode 主循环成本
- `flush_stream` 虽然不是总热点第一，但它解释了为什么 output 没有看起来那么轻

读真实 profiler 时也一样。不要先被工具界面牵着走，先问自己：当前 top hotspot 落在 `prefill`、`decode`，还是服务输出。

## 再往工具视图走一步

当前教程基线还可以导出一份简化的 Chrome trace 风格结果。它不是完整 profiler 导出物，但已经接近真实工具常见的时间轴事件格式：

```json
{
  "traceEvents": [
    {"name": "thread_name", "ph": "M", "pid": 1, "tid": 0, "args": {"name": "host-thread"}},
    {"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "compute-stream"}},
    {"name": "run_prefill_attention", "cat": "prefill", "ph": "X", "pid": 1, "tid": 1, "ts": 130.00, "dur": 140.00},
    {"name": "decode_step", "cat": "decode", "ph": "X", "pid": 1, "tid": 1, "ts": 300.00, "dur": 90.00},
    {"name": "sync_decode_result", "cat": "sync", "ph": "i", "pid": 1, "tid": 0, "ts": 450.00},
    {"name": "flush_stream", "cat": "output", "ph": "X", "pid": 1, "tid": 0, "ts": 530.00, "dur": 50.00}
  ]
}
```

这一层的意义不是教读者记字段，而是建立一个判断：文本摘要、hotspot 表、时间轴视图，本质上是在表达同一组证据，只是观察角度不同。

这里多出来的两类信息很关键：

- `tid`：帮助你区分 host 侧和 compute 侧事件
- `sync`：帮助你看到性能不是只花在算子上，也可能被等待点打断

这已经足够支撑第 8 章的教程目标，不需要继续实现完整 profiler。
