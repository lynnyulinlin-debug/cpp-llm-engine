# cpp-llm-engine

用 C++ 构建 LLM 推理引擎

_A Practical Tutorial for Building a C++ LLM Inference Engine_

[![CPU and docs CI](https://github.com/lynnyulinlin-debug/cpp-llm-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/lynnyulinlin-debug/cpp-llm-engine/actions/workflows/ci.yml)

这是一套中文优先、以 C++ 为实现主线、CPU 可完整学习、GPU 可选加速的推理引擎工程教程。目标是从真实模型基线出发，逐步构建可以自行修改和验证的推理服务框架。

主图从上到下展示学习阶段、代码骨架和能力扩展。CPU 主线负责形成完整推理服务，CUDA 替换执行后端，多模态扩展输入路径，成熟框架用于核对工程策略。

![cpp-llm-engine 教程主图](source/assets/diagrams/tutorial-concept-map.svg)

## 开始学习

从 [`Chapter 00`](source/chapter00_quickstart/intro.md) 跑通 TinyLlama 真实推理，再按 Chapter 01-09 构建 CPU 推理服务。C++ 或 CMake 基础不足时按需学习 [`Foundation`](source/foundation/README.md)，完整命令和路线见 [USAGE.md](USAGE.md)。

CPU v1 已形成从 GGUF、Decoder、Runtime 到 HTTP/SSE 的可运行闭环；希望按阶段复现时使用[第一期 CPU 学习计划](source/guides/phase1_cpu_learning_plan.md)，完成后再选择 CUDA 或多模态扩展。

## 教程定位与特点

教程沿“真实基线 → 数据与算法契约 → C++ 实现 → 验证证据 → 成熟框架对照”推进。Python、NumPy 和外部项目只承担解释或独立参考；可运行范围聚焦单机 CPU、可选单消费级 NVIDIA GPU 和多模态输入扩展，不宣称多 GPU、分布式或生产级多租户能力。

| 教程特点 | 学习者获得什么 | 可信度约束 |
| --- | --- | --- |
| 真实模型纵向闭环 | 从 GGUF、Tokenizer、Embedding、Decoder 到 Runtime 和 ServiceCore | smoke、真实模型和服务能力分别说明，不互相替代 |
| C++ 主线、CPU 可完整学习 | 无 GPU 也能理解并运行完整引擎主线，CUDA 和多模态复用上层契约 | 可选能力使用独立构建和验证入口 |
| Correctness-first | 每项实现附带单测、golden、benchmark 或 profiling 证据 | 正确性失败时不讨论性能收益 |
| **章节级框架对照卡** | 把本章对象映射到 `llama.cpp`、vLLM、SGLang、TensorRT-LLM 等真实工程 | 区分概念相似、接口映射和实现等价，不宣称未验证能力 |
| 教学与生产分层 | 识别“已实现”“教学设计”“生产级参照”三种深度 | 生产组件只给约束与源码坐标，不写成仓库现成功能 |

## 建设时间线

更新时间：**2026-10-26**。时间线只记录已经形成代码和验证入口的里程碑；详细实验条件见[实验资产表](source/reference/experiment_assets.md)。

| 时间 | 里程碑 | 状态 | 可验证产物 |
| --- | --- | --- | --- |
| 2026-08 | 教程骨架与 Foundation | 完成 | 章节主线、基础题库、benchmark trace |
| 2026-10 | CPU v1 | 完成 | TinyLlama 22 层 Decoder、Runtime、HTTP/SSE 和统一验收 |
| 2026-10 | GPU v1 | 建设中 | CUDA 资源、量化 Linear、RoPE、KV、GQA 和合成 Decoder；下一步接真实模型 |
| 2026-10 | 多模态 CPU v1 | 完成 | SmolVLM Vision、Projector、Decoder、ServiceCore 和 profiling golden |
| 2026-10 | 工程验证 | 完成 | 默认 CPU 回归、文档检查和 GitHub Actions CI |

## 基线与实现边界

模型优先选择单机可运行、结构具有代表性且能由外部实现复核的资产。教程自行实现执行、状态和资源管理；模型专属兼容细节与 NumPy 镜像只提供外部参考。

| 对象 | 采用方式 | 自研与复用边界 | 主要证据 |
| --- | --- | --- | --- |
| TinyLlama 1.1B Q4_K_M | CPU 语言主线与 GPU 首个目标 | 复用 `llama.cpp` tokenizer；自研 GGUF 对象、Embedding、Decoder、Runtime 和 Serving | token、logits、HTTP/SSE golden |
| SmolVLM 500M Q8_0 + mmproj | 多模态 CPU 主线 | 复用模型格式与 mtmd baseline；自研图像管线、Vision、Projector 和图文执行 | C++/NumPy checksum、服务与 profiling golden |
| `llama.cpp` / NumPy | 独立 reference | 不进入默认 C++ Runtime，不作为自研 Engine 的替代品 | hash、shape、top-k 和输出序列 |
| 固定 prompt、图片与合成 Tensor | 小型测试 fixture | 只验证接口、数值和生命周期，不代表模型质量或数据集覆盖 | 默认 CTest 与跨实现对照 |

## 章节入口

主线顺序固定为：

```text
Foundation -> Chapter00 -> Chapter01 -> Chapter02 -> Chapter03 -> Chapter04
    -> Chapter05 -> Chapter06 -> Chapter07 -> Chapter08 -> Chapter09
```

Chapter09 虽然放在总主线末尾，但服务化专题可以在完成 Chapter05 后提前阅读。

| 模块 | 定位 | 主要内容 | 完成后的产物 |
| --- | --- | --- | --- |
| [`foundation`](source/foundation/README.md) | 前置准备 | C++、构建、CUDA、调试、测试基础 | 基础练习和最小工程 |
| [`chapter00_quickstart`](source/chapter00_quickstart/intro.md) | 起跑线 | 第一次真实推理 | llama.cpp + TinyLlama CPU 输出 |
| [`chapter01_tensor_loader`](source/chapter01_tensor_loader/intro.md) | 输入层 | 权重加载、`Tensor` 与真实 Model 构建 | 可检查的 tokenizer、metadata、权重和 Model |
| [`chapter02_tensor_ops`](source/chapter02_tensor_ops/intro.md) | 张量语义层 | 视图语义、共享存储、`reshape`、`slice` | 共享 Storage 的 Tensor view |
| [`chapter03_matmul`](source/chapter03_matmul/intro.md) | 算子边界层 | `MatMul`、后端拆分、参考实现 | 可验证的 CPU MatMul backend |
| [`chapter04_attention`](source/chapter04_attention/intro.md) | 结构层 | Attention、FFN、DecoderLayer | 可测试的单层 Decoder |
| [`chapter05_generation`](source/chapter05_generation/intro.md) | Generation Runtime | Prefill / Decode、Request、KV Cache、Scheduler、Streaming | 可推进单请求和多请求的 CPU Runtime |
| [`chapter06_quantization`](source/chapter06_quantization/intro.md) | 表示与成本层 | 量化路径和精度权衡 | Q4_K/Q6_K Linear 正确性与性能对照 |
| [`chapter07_benchmark`](source/chapter07_benchmark/intro.md) | 验证层 | 正确性验证和性能比较 | 可复现的 benchmark JSON |
| [`chapter08_profiling`](source/chapter08_profiling/intro.md) | 归因与优化层 | Profiling 证据和 A/B 对比 | 阶段、trace、hotspot 和优化决策 |
| [`chapter09_serving`](source/chapter09_serving/intro.md) | 服务扩展层 | C++ Service Core、动态 batch、Block KV、HTTP/SSE | 可测试的最小 CPU 推理服务 |
| [`code/cuda`](code/cuda/README.md) | GPU 执行扩展 | 设备资源、量化 Linear、RoPE、连续 KV、GQA、合成 Decoder | CPU/CUDA 数值对照与设备 CTest |
| [`code/multimodal`](code/multimodal/README.md) | 多模态输入扩展 | 动态切图、Vision Encoder、Projector、embedding 拼接 | SmolVLM CPU Decoder、ServiceCore 与 profiling golden |
| [`guides`](source/guides/README.md) | 工程参考层 | 构建、环境、迁移和工程方法 | 按需使用的操作指南 |
| [`reference`](source/reference/README.md) | 解释层 | 术语、生态对照、概念映射 | 跨章节查询入口 |

## 进阶优化路线

完成 Chapter07 的正确性与 benchmark 基线后，可以沿下表继续学习性能专题；这些内容不替代 Chapter08-09 的 profiling 和服务主线。

| Task | 专题方向 | 关注点 |
| --- | --- | --- |
| 1 | 推理结构地基 | `MHA` / `GQA`、`LLaMA3 Block` |
| 2 | Attention 访存瓶颈与 FlashAttention | 访存模型、`SRAM` 优化、`FlashAttention` |
| 3 | 解码算法 | `Decoding Strategies`、`Speculative Decoding`、`Multi-Token Decoding`、`Decode Scheduling` |
| 4 | `KV Cache` 与推理服务内存管理 | `PagedAttention`、`RadixAttention`、`Prefix Caching`、`Chunked Prefill`、`KV Cache Scheduling` |
| 5 | 量化推理与部署 | `INT4 / INT8`、`W8A16`、`GPTQ / AWQ`、`FP8`、`KV Cache Quantization` |
| 6 | 扩展选读与专题复盘 | `Inference Performance Comparison` |

## 使用入口

学习步骤、环境配置、贡献规则和查询资产分别维护，README 不复制这些文档的具体命令和细节。

| 需要解决的问题 | 入口 |
| --- | --- |
| 如何学习和运行 | [USAGE.md](USAGE.md) |
| CPU/GPU 环境与代码架构 | [docs/guide.md](docs/guide.md) |
| 如何修改和提交 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 章节代码、实验与框架坐标 | [Reference](source/reference/README.md) |
| 术语与公式 | [术语表](source/reference/glossary.md) · [公式表](source/reference/formulas.md) |

## 验证与 CI

默认 CPU 回归不依赖 GGUF、网络下载或 GPU；真实模型、transport、性能和 CUDA 使用各自入口补充证据。GitHub Actions 会在 `main` push 和 pull request 上执行文档检查与默认回归。

```bash
npm install
npm run docs:check-style
npm run docs:check-links
npm run test:regression
```

![教程验证证据图](source/assets/diagrams/test-pipeline.svg)
