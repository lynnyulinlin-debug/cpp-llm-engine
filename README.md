# cpp-llm-engine

用 C++ 构建 LLM 推理引擎

_A Practical Tutorial for Building a C++ LLM Inference Engine_

这是一个面向中文学习者优先的工程教程仓库。教程入口和阅读路径见下方各节。

## 项目简介

- 中文优先，英文入口逐步补齐
- 主要内容是从 `foundation` 到 `chapter07` 的 C++ LLM 推理主线，外加 `guides`、`reference`、`maintenance` 三层支撑
- 特点是按工程问题推进，强调可跑、可验证、可回看
- 适合想从零看懂 C++ 推理引擎，并继续把它推进到可解释、可验证、可维护的读者

这套教程的定位，是一条面向 C++ 的推理引擎原理、最小实现与验证路径。它不把自己写成某个单一开源项目的替代说明，而是把 `llama.cpp`、`vLLM`、`SGLang`、`TensorRT-LLM`、`Ollama` 等工程坐标作为对照对象，用来帮助读者理解真实推理系统在不同场景下如何分层、如何调度、如何做内存管理，以及为什么会形成不同的工程取舍。

## 教程地图

| 模块 | 定位 | 主要内容 | 状态 |
| --- | --- | --- | --- |
| [`foundation`](source/foundation/README.md) | 前置准备 | C++、构建、CUDA、调试、测试基础 | 完成 |
| [`chapter00_quickstart`](source/chapter00_quickstart/intro.md) | 起跑线 | 第一次真实推理 | 完成 |
| [`chapter01_tensor_loader`](source/chapter01_tensor_loader/intro.md) | 输入层 | 权重加载与 `Tensor` 抽象 | 基本完成 |
| [`chapter02_tensor_ops`](source/chapter02_tensor_ops/intro.md) | 张量语义层 | 视图语义、共享存储、`reshape`、`slice` | 基本完成 |
| [`chapter03_matmul`](source/chapter03_matmul/intro.md) | 算子边界层 | `MatMul`、后端拆分、参考实现 | 基本完成 |
| [`chapter04_attention`](source/chapter04_attention/intro.md) | 结构层 | attention、FFN、decoder layer | 基本完成 |
| [`chapter05_generation`](source/chapter05_generation/intro.md) | 运行时层 | token-by-token generation、KV cache、sampler、scheduler | 基本完成 |
| [`chapter06_quantization`](source/chapter06_quantization/intro.md) | 表示与成本层 | 量化路径和精度权衡 | 基本完成 |
| [`chapter07_benchmark`](source/chapter07_benchmark/intro.md) | 验证层 | 正确性验证、性能比较、证据收口 | 基本完成 |
| [`chapter08_profiling`](source/chapter08_profiling/intro.md) | 归因与优化层 | profiling、A/B 对比、优化闭环 | 初稿 |
| [`guides`](source/guides/README.md) | 工程参考层 | 构建、环境、迁移和工程方法 | 完成 |
| [`reference`](source/reference/README.md) | 解释层 | 术语、生态对照、概念映射 | 待补 |
| [`maintenance`](source/maintenance/README.md) | 维护层 | 写作规范、模板、长期收口 | 待补 |

## 使用文档

如果你要先搭环境、再选学习路径，先看 [docs/guide.md](docs/guide.md)。这页会把阅读顺序、CPU / GPU 分界线，以及本地和云端的环境建议说明清楚。

## 三条学习路径

| 路径 | 适合对象 | 阅读顺序 | 最小产物与验证 |
| --- | --- | --- | --- |
| 初学者路径 | C++、CMake 或推理基础不完整 | `foundation` -> `chapter00` -> `chapter01` -> `chapter02` | 跑通第一次推理；执行 `npm run test:regression` |
| C++ 工程路径 | 能写 C++，希望理解对象、算子和 runtime | `foundation/FQ01-FQ10` -> `chapter01` -> `chapter03` -> `chapter05` -> `chapter07` | 完成可测试 Tensor、算子和 generation runtime |
| 推理引擎进阶路径 | 已有 C++/GPU 基础，关注系统设计和性能 | `chapter05` -> `reference` -> `chapter07` -> `chapter08` | 能解释调度、KV Cache、回归和 profiling 证据 |

每条路径都可以先走 CPU-first；只有进入 `chapter03` 或 FQ10 CUDA 选修时，才需要 GPU 环境。

## 进阶优化路线

这条路线面向已经读完主线、想继续深入推理优化的读者。它不是主线替代，而是主线之后的专题延伸。

| Task | 专题方向 | 关注点 |
| --- | --- | --- |
| 1 | 推理结构地基 | `MHA` / `GQA`、`LLaMA3 Block` |
| 2 | Attention 访存瓶颈与 FlashAttention | 访存模型、`SRAM` 优化、`FlashAttention` |
| 3 | 解码算法 | `Decoding Strategies`、`Speculative Decoding`、`Multi-Token Decoding`、`Decode Scheduling` |
| 4 | `KV Cache` 与推理服务内存管理 | `PagedAttention`、`RadixAttention`、`Prefix Caching`、`Chunked Prefill`、`KV Cache Scheduling` |
| 5 | 量化推理与部署 | `INT4 / INT8`、`W8A16`、`GPTQ / AWQ`、`FP8`、`KV Cache Quantization` |
| 6 | 扩展选读与专题复盘 | `Inference Performance Comparison` |

这条路线适合在 `chapter07` 之后继续看，也适合把主线中的 `attention`、`generation`、`quantization` 和 `benchmark` 再往优化视角拉深。

## 相关入口

- [docs/README.md](docs/README.md)
- [docs/README.en.md](docs/README.en.md)
- [docs/guide.md](docs/guide.md)
- [docs/index.md](/)
- [source/reference/README.md](source/reference/README.md)
- [source/reference/ecosystem/README.md](source/reference/ecosystem/README.md)
- [source/maintenance/README.md](source/maintenance/README.md)

## 构建与检查

```bash
npm install
npm run docs:check-links
npm run docs:build
npm run docs:preview
npm run test:regression
```

如果你的环境需要指定镜像：

```bash
npm install --registry=https://registry.npmjs.org
```
