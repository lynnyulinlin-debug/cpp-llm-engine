# Reference 设计参考

这里放的是概念解释、对比分析、术语索引和继续阅读材料。它面向“想理解设计”的读者，而不是正在改代码的读者。

## 内容范围

- [framework_comparison.md](./framework_comparison.md)：训练框架、推理框架和本教程的关系
- [inference_engine_systems.md](./inference_engine_systems.md)：推理引擎运行时、内存、profiling、输入链路和服务化补强图
- [execution_layers.md](./execution_layers.md)：算子、Backend、Executor 与 Runtime 的分层
- [memory_management.md](./memory_management.md)：Tensor、workspace、KV Cache 和 staging 的资源生命周期
- [runtime_systems.md](./runtime_systems.md)：推理引擎运行时边界
- [profiling.md](./profiling.md)：benchmark 之后的瓶颈解释
- [input_pipeline.md](./input_pipeline.md)：推理引擎输入链路
- [service_basics.md](./service_basics.md)：推理引擎服务化基础
- [regression_testing.md](./regression_testing.md)：回归测试和版本化基线
- [glossary.md](./glossary.md)：核心术语表
- [faq.md](./faq.md)：常见问题
- [further_reading.md](./further_reading.md)：主线之外继续看什么
- [ecosystem/README.md](./ecosystem/README.md)：外部推理生态对照

## 怎么用

- 理解概念、术语或生态定位时，可以从这里查
- 编写代码时，这里不是第一入口
- 这些页面以解释和对比为主，不展开成主线正文
