# 用 C++ 构建 LLM 推理引擎

_A Practical Tutorial for Building a C++ LLM Inference Engine_

这是整套教程给学习者看的中文主入口。若你更习惯英文导航，可以先看 [English entry](/en/README)。

## 这套教程的定位

这不是“讲概念最多”的资料集，而是一条可顺着读、顺着跑、顺着改的工程路径。它面向的是想从零看懂 C++ 推理引擎、并把“能跑”继续推进到“能解释、能验证、能扩展”的读者。

中文主线已经基本完成，当前重点转向英文正文、残余句式清理，以及参考层和维护层的持续补齐。

- 从 `foundation` 补基础底座
- 从 `chapter00_quickstart` 跑通真实基线
- 从 `Tensor`、`MatMul`、`Attention` 走到 `Generation`
- 再用 `Quantization` 和 `Benchmark` 收口
- 主线中文正文已收束到“导读 / 机制 / 收口”三层，代码直达也已统一到具体实现文件
- 以 `foundation`、主线章节、`guides` / `reference` / `maintenance` 三层结构组织内容
- 用章节导读、代码索引和收口页把“读文档”和“看代码”连起来
- 保持中文优先，同时保留英文入口和逐步扩展能力
- 允许 CPU-first 学习路径先跑通，再在需要时接入 CUDA / GPU 后端

## 教程主线

这套教程不是按“主题合集”组织的，而是按一条推理引擎主线往前推进。

| 阶段 | 主要职责 | 读完后你应该能理解什么 |
| --- | --- | --- |
| `chapter01` - `chapter03` | 输入、张量语义、算子边界 | 权重怎样进入程序，张量怎样表达视图，计算怎样变成稳定边界 |
| `chapter04` - `chapter05` | 结构与运行时 | 单层 decoder 怎样站稳，decoder 怎样进入生成 runtime |
| `chapter06` | 表示与成本 | 量化怎样降低资源成本，同时保住可接受精度 |
| `chapter07` | 验证与证据 | 正确性、性能和 benchmark 结果怎样被系统地验证和比较 |

`chapter07` 之后，主线不再继续扩成新的教学章节，而是转入解释层和工程层：

- `reference` 负责概念解释、术语和生态对照
- `guides` 负责构建、环境、迁移和工程方法
- `maintenance` 负责站点维护、写作规范和长期收口

如果你想先把“指挥台 / 发动机”这层关系讲清楚，先看 [training vs inference frameworks](/reference/framework_comparison)。这页会把 Python 框架、C++ 引擎、以及它们在系统里的地位和结构组成放在一张图里看。

这意味着这套教程的收束方式很明确：先把一个推理引擎做出来，再把它验证清楚，最后把它放回真实生态里对照和维护。

## 和常见推理引擎的对照

- `llama.cpp`：本地基线，适合先跑起来，对应 `chapter00_quickstart` 和 `chapter01_tensor_loader`
- `vLLM`：serving、调度和 KV cache，对应 `chapter05_generation` 和 `chapter07_benchmark`
- `TensorRT-LLM`：GPU engine、runtime 和量化部署，对应 `chapter03_matmul`、`chapter04_attention`、`chapter06_quantization`
- `Ollama`：本地交付和打包，对应 `chapter00_quickstart` 和 `chapter05_generation`
- `Gemma`：模型家族参考和验证对象，对应 `chapter01_tensor_loader`、`chapter04_attention`、`chapter06_quantization`、`chapter07_benchmark`
- `SGLang`：结构化 serving 和流程编排，对应 `chapter05_generation` 和 `chapter07_benchmark`

```mermaid
flowchart LR
    A[本教程主线] --> L[llama.cpp]
    A --> V[vLLM]
    A --> S[SGLang]
    A --> T[TensorRT-LLM]
    A --> O[Ollama]
    A --> G[Gemma]

    L -. 本地基线 .-> C0[chapter00_quickstart]
    L -. 权重加载 .-> C1[chapter01_tensor_loader]
    V -. 吞吐 / 调度 .-> C5[chapter05_generation]
    S -. 流程编排 .-> C5
    T -. engine / runtime .-> C3[chapter03_matmul]
    T -. engine / quantization .-> C4[chapter04_attention]
    T -. engine / quantization .-> C6[chapter06_quantization]
    O -. 打包交付 .-> C0
    G -. 模型行为 .-> C1
```

## 章节总览

| 顺序 | 模块 | 主要内容 | 英文覆盖状态 |
| --- | --- | --- | --- |
| 0 | [`foundation`](/foundation/README) | C++、构建、CUDA、调试与测试基础 | 英文正文已覆盖 |
| 1 | [`chapter00_quickstart`](/chapter00_quickstart/intro) | 先跑通第一次真实推理 | 英文导读 / 基本完成 |
| 2 | [`chapter01_tensor_loader`](/chapter01_tensor_loader/intro) | 权重加载与 `Tensor` 抽象 | 英文导读 / 基本完成 |
| 3 | [`chapter02_tensor_ops`](/chapter02_tensor_ops/intro) | 视图语义、共享存储、`reshape`、`slice` | 英文导读 / 基本完成 |
| 4 | [`chapter03_matmul`](/chapter03_matmul/intro) | `MatMul` 算子、后端拆分、参考实现 | 英文导读 / 基本完成 |
| 5 | [`chapter04_attention`](/chapter04_attention/intro) | attention、FFN、decoder layer | 英文导读 / 基本完成 |
| 6 | [`chapter05_generation`](/chapter05_generation/intro) | token-by-token generation、KV cache、sampler | 英文导读 / 基本完成 |
| 7 | [`chapter06_quantization`](/chapter06_quantization/intro) | 量化推理路径和精度权衡 | 英文导读 / 基本完成 |
| 8 | [`chapter07_benchmark`](/chapter07_benchmark/intro) | 正确性验证、性能比较、证据收口 | 英文导读 / 基本完成 |
| 9 | [`guides`](/guides/README) | 开发参考、构建、环境、迁移 | 英文入口已覆盖 |
| 10 | [`reference`](/reference/README) | 设计参考、术语、生态对照 | 部分完成 |
| 11 | [`maintenance`](/maintenance/README) | 教程维护、写作规范、模板 | 部分完成 |

## 适合谁

- 想从零看懂 C++ 推理引擎的人
- 想把“能跑”继续推进到“能解释、能验证、能扩展”的人
- 想对比 `llama.cpp`、`vLLM`、`SGLang`、`TensorRT-LLM`、`Ollama` 的工程思路的人

## 读法建议

1. 先看 [USAGE.md](/guide)
2. 再看 [docs/README.md](/README)
3. 基础薄弱就先看 [source/foundation/README.md](/foundation/README)
4. 想先跑结果就从 `chapter00_quickstart` 开始
5. 想做工程判断就重点看 `chapter05_generation` 和 `chapter07_benchmark`

## CPU / GPU 分界线

- `chapter00`、`chapter01`、`chapter02`：先用 CPU 路径就能学和验证
- `chapter03_matmul`：开始把 GPU 作为明确后端引入主线
- 第一处具体 GPU 实现：[`chapter03_matmul/04_cublas_implement.md`](/chapter03_matmul/04_cublas_implement)
- 开发参考入口：[`guides/README.md`](/guides/README)
- 设计参考入口：[`reference/README.md`](/reference/README)
- 维护文档入口：[`maintenance/README.md`](/maintenance/README)

## 本地预览

先安装依赖，再构建或预览：

```bash
npm install
npm run docs:check-links
npm run docs:build
npm run docs:preview
```

本地开发预览：

```bash
npm run docs:dev
```

如果你的环境需要指定镜像：

```bash
npm install --registry=https://registry.npmjs.org
```

## 相关入口

- [docs/README.md](/README)
- [docs/README.en.md](/en/README)
- [docs/guide.md](/guide)
- [docs/index.md](/)
- [source/reference/README.md](/reference/README)
- [source/maintenance/README.md](/maintenance/README)

## English coverage status

| Page | Status | Route |
| --- | --- | --- |
| `docs/index.md` | done | `/en/` |
| `docs/README.md` | done | `/en/README` |
| `docs/guide.md` | done | `/en/guide` |
| `source/foundation/README.md` and foundation pages | done | `/en/foundation/README` and `/en/foundation/*` |
| chapter intro / summary pages | done | `/en/chapterXX/intro` and `/en/chapterXX/summary` |
| `guides` / `reference` / `maintenance` entry pages | done | `/en/guides/README`, `/en/reference/README`, `/en/maintenance/README` |
