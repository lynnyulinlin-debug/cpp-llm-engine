# cpp-llm-engine

用 C++ 构建 LLM 推理引擎

_A Practical Tutorial for Building a C++ LLM Inference Engine_

这是一个面向中文学习者优先的工程教程仓库。教程入口和阅读路径见下方各节。

## 项目简介

- 中文优先，英文入口逐步补齐
- 主要内容是从 `foundation` 到 `chapter07` 的 C++ LLM 推理主线，外加 `guides`、`reference`、`maintenance` 三层支撑
- 特点是按工程问题推进，强调可跑、可验证、可回看
- 适合想从零看懂 C++ 推理引擎，并继续把它推进到可解释、可验证、可维护的读者

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
| [`guides`](source/guides/README.md) | 工程参考层 | 构建、环境、迁移和工程方法 | 完成 |
| [`reference`](source/reference/README.md) | 解释层 | 术语、生态对照、概念映射 | 待补 |
| [`maintenance`](source/maintenance/README.md) | 维护层 | 写作规范、模板、长期收口 | 待补 |

## 相关入口

- [docs/README.md](docs/README.md)
- [docs/README.en.md](docs/README.en.md)
- [docs/guide.md](docs/guide.md)
- [docs/index.md](/)
- [source/reference/README.md](source/reference/README.md)
- [source/maintenance/README.md](source/maintenance/README.md)

## 构建与检查

```bash
npm install
npm run docs:check-links
npm run docs:build
npm run docs:preview
```

如果你的环境需要指定镜像：

```bash
npm install --registry=https://registry.npmjs.org
```
