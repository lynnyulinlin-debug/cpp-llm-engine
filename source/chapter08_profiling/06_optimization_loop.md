# 08.6 Optimization Loop 优化闭环

性能优化不是不断加技巧，而是一条闭环：提出假设、修改实现、重新测量、解释结果、决定保留或回滚。

**阅读入口**

- 前置阅读：[`08.5 A/B Experiment`](./05_ab_experiment.md)、[`07.5 Optimization Tips`](../chapter07_benchmark/05_optimization_tips.md)
- 源码直达：<a href="../_code/chapter07_benchmark/src/benchmark.cpp"><code>benchmark.cpp</code></a>、<a href="../_code/chapter05_generation/src/runtime.cpp"><code>runtime.cpp</code></a>、<a href="../_code/chapter05_generation/src/kv_cache.cpp"><code>kv_cache.cpp</code></a>
- 相关阅读：[`推理引擎系统补强图`](../reference/inference_engine_systems.md)、[`08.7 vLLM / nano-vLLM Profiling 阅读桥`](./07_vllm_nano_vllm_profiling_bridge.md)

## 优化步骤

```text
benchmark -> hypothesis -> profiling -> change -> benchmark -> decision
```

每一步只回答一个问题：

- benchmark：问题是否真实存在？
- hypothesis：你认为瓶颈在哪？
- profiling：证据是否支持这个判断？
- change：这次只改什么？
- benchmark：改动是否产生可复查收益？
- decision：保留、回滚，还是继续拆？

### 保留优化的条件

一个优化至少要同时满足三件事：

- 正确性没有退化
- 目标指标有稳定改善
- 复杂度和维护成本可以接受

如果只满足第二项，它更像一次实验结果，还不是可以进入主线的工程改动。

### 什么时候不要继续优化

下面几种情况应该先停下来，而不是继续叠优化：

- benchmark 条件没有固定，结果不可复现
- 正确性验证没有通过
- profiling 证据和优化方向对不上
- 改动收益只出现在单次运行里，重复测量不稳定
- 复杂度明显增加，但目标指标改善很小

这不是保守，而是工程边界。推理引擎优化最终要进入长期维护，不能只看一次跑分。

### 用当前教程代码走一遍闭环

拿 `chapter07_benchmark_demo` 的真实输出，可以先走一遍不引入任何新优化技巧的最小闭环：

1. `benchmark`：当前 baseline 是 `4000 items/s`，并且 `prefill=0.30ms`、`decode=0.15ms`
2. `hypothesis`：当前阶段占比里 `prefill=50%`、`decode=25%`，如果后续长 prompt 场景变慢，优先怀疑 `prefill` 路径而不是输出阶段
3. `profiling`：先围绕 `runtime.cpp`、`benchmark.cpp` 补阶段计时，而不是一开始就猜 kernel
4. `change`：只增加更细的阶段计时，先不改 scheduler、cache 和 kernel
5. `benchmark`：确认新增计时后，基线结果仍能稳定复现
6. `decision`：如果计时开销可接受，就保留这次可观测性改动

这条示例故意很小，因为教程当前要先建立“证据链”习惯，而不是堆优化技巧。

## 结构化输出

当阶段计时、trace 事件和 hotspot 表都已经出现之后，再往前一步就是把它们收进同一份结构化结果。原因很直接：

- 文本适合阅读
- JSON 适合复查、对比和后续工具消费

也就是说，优化闭环的终点不只是“看懂一次结果”，而是“把这次结果保存成后面还能比较的证据”。

对当前教程来说，这意味着一次完整闭环至少应该留下三样东西：

- 一份人能直接读的 benchmark / A/B 摘要
- 一份事件级 trace
- 一份结构化 JSON 结果，方便后续比较和工具消费

如果继续贴近真实工程，再多保留一份“时间轴友好”的 trace 视图也很有价值。它不一定是完整工具导出，但至少要让后来的人能快速看出事件先后顺序和重叠关系。
