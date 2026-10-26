# Reference 设计参考

这里放的是概念解释、对比分析、术语索引和继续阅读材料。它面向“想理解设计”的读者，而不是正在改代码的读者。

## 内容范围

- [framework_comparison.md](./framework_comparison.md)：训练框架、推理引擎、推理服务和本教程的系统位置
- [inference_engine_systems.md](./inference_engine_systems.md)：推理引擎运行时、内存、profiling、输入链路和服务化补强图
- [execution_layers.md](./execution_layers.md)：算子、Backend、Executor 与 Runtime 的分层
- [chapter_assets.md](./chapter_assets.md)：各章正文、代码、图、公式和验证入口
- [experiment_assets.md](./experiment_assets.md)：性能实验、正确性 golden、结果文件和复现入口
- [data_structures_algorithms.md](./data_structures_algorithms.md)：跨章节数据结构、算法、代码入口和验证产物总表
- [cpu_backend_architecture.md](./cpu_backend_architecture.md)：CPU 主线、公共契约和 GPU 扩展边界
- [cpu_gpu_architecture.md](./cpu_gpu_architecture.md)：CPU 主线、GPU 扩展和证据闭环的统一架构
- [memory_management.md](./memory_management.md)：Tensor、workspace、KV Cache 和 staging 的资源生命周期
- [runtime_systems.md](./runtime_systems.md)：推理引擎运行时边界
- [profiling.md](./profiling.md)：benchmark 之后的瓶颈解释
- [input_pipeline.md](./input_pipeline.md)：推理引擎输入链路
- [multimodal_inference.md](./multimodal_inference.md)：图片动态切图、视觉编码和 LLM 接入链路
- [service_basics.md](./service_basics.md)：推理引擎服务化基础
- [regression_testing.md](./regression_testing.md)：回归测试和版本化基线
- [formulas.md](./formulas.md)：Tensor、模型、Runtime、性能和多模态公式索引
- [glossary.md](./glossary.md)：核心术语表
- [faq.md](./faq.md)：常见问题
- [further_reading.md](./further_reading.md)：主线之外继续看什么
- [ecosystem/README.md](./ecosystem/README.md)：按工程问题选择外部项目卡片

## 怎么用

- 理解概念、术语或生态定位时，可以从这里查
- 编写代码时，这里不是第一入口
- 这些页面以解释和对比为主，不展开成主线正文

## 图片资源

图片资源统一放在 `source/assets/diagrams/`，正文使用相对路径 `../assets/diagrams/<name>.svg` 引用。`docs/assets` 是站点源码树的映射，`docs/.vitepress/public/diagrams` 仅保留旧站点 URL 的兼容映射。提交文档时必须同时提交对应 SVG；`npm run docs:check-links` 会检查源码相对路径、生成 HTML 中的 `href` 和图片 `src`，避免只提交正文而遗漏图片资源。
