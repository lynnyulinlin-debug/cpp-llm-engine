# 01.9 Loader 框架对照：从 GGUF 到可执行模型

本章把加载过程拆成“解析文件、归一化配置、构造 Tensor、装配 Model、创建请求 Context”。成熟框架也要完成这些职责，只是支持更多格式、设备和并行策略。

## 职责映射

| 教程对象 | `llama.cpp` | vLLM | 阅读重点 |
| --- | --- | --- | --- |
| `GgufReader`、`GgufTensorLoader` | GGUF reader、model loader | checkpoint loader | 文件索引何时转换为权重对象 |
| `ModelMetadataAdapter`、`ModelConfig` | architecture metadata、hparams | model config、loader config | 外部字段怎样归一化并校验 |
| `ModelBuilder`、layer mapping | model tensor creation/loading | 参数名映射和 weight loading | 文件中的 tensor 名怎样绑定到模型层 |
| `RealLlamaCpuModel` | 共享模型权重 | worker/model runner 中的模型实例 | 哪些资源跨请求共享 |
| `RealLlamaCpuContext` | per-context inference state | request/runtime state | 哪些状态必须按请求隔离 |

## 教学实现与生产实现

本章用单个 TinyLlama GGUF 固定输入，目的是让格式边界和对象生命周期可见。

| 本章可验证对象 | 成熟系统继续增加 | 学习顺序 |
| --- | --- | --- |
| GGUF header、metadata、tensor index 和 payload offset | SafeTensors、分片 checkpoint、远程下载 | 先固定一种真实格式 |
| TinyLlama 配置和 tensor 名映射 | 多模型架构注册表 | 先验证 Llama 主线 |
| CPU 权重对象和请求 Context | GPU placement、tensor parallel、pipeline parallel | 后端与分布式不属于加载入门闭环 |
| 明确失败的 shape/dtype/metadata 校验 | lazy loading、mmap 策略和权重缓存 | 先保证错误可观察 |

## 源码阅读问题

1. tensor 名称在什么时候从文件 schema 转换为模型层参数？
2. 模型权重与请求级 KV/position 状态分别由谁持有？
3. 增加一种 checkpoint 格式时，哪些内部对象和校验不应改变？

| 项目卡 | 在本章中的用途 |
| --- | --- |
| [llama.cpp](../reference/ecosystem/llama_cpp.md) | GGUF Loader、模型对象和 Context |
| [vLLM](../reference/ecosystem/vllm.md) | 多格式 checkpoint Loader 与模型实例 |
