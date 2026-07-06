# TensorRT-LLM GPU 推理

## 解决什么问题

`TensorRT-LLM` 面向的是 NVIDIA GPU 上的高性能 LLM 推理。它适合用在这样的问题上：如何把一个模型变成经过优化的 engine 和 runtime，并且具备面向生产的吞吐能力。

## 工程策略

- 把模型定义和 engine 构建、运行时执行分开
- 优化 attention、KV cache、批处理和量化路径
- 面向大规模部署提供 runtime 和 serving 相关的基础模块

## 什么时候选它

如果你的问题是高性能 NVIDIA GPU 推理，就优先看 `TensorRT-LLM`：

- engine build 和 runtime 分层很重要
- kernel 性能很重要
- 量化本身就是部署策略的一部分
- 系统需要一条面向生产的 GPU 执行路径

## 对应教程章节

`TensorRT-LLM` 主要对应教程里的这些章节：

- `chapter03_matmul`
- `chapter04_attention`
- `chapter05_generation`
- `chapter06_quantization`
- `chapter07_benchmark`
