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

## 源码入口

下面入口来自框架官方仓库。先追一条权重从文件名到模型参数的路径，不要从整个仓库目录开始读。

| 框架 | 官方入口 | 带着什么问题读 |
| --- | --- | --- |
| `llama.cpp` | [`llama-model-loader.h`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.h)、[`llama-model-loader.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp) | GGUF metadata、tensor index 和 payload 在哪里校验 |
| `llama.cpp` | [`llama-model.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp) | loader 输出怎样变成可执行模型 |
| vLLM | [`model_loader`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader) | loader 怎样选择格式并把权重交给模型实现 |

## 教程保留与省略

本章用单个 TinyLlama GGUF 固定输入，目的是让格式边界和对象生命周期可见。

| 已实现 | 暂不实现 | 省略原因 |
| --- | --- | --- |
| GGUF header、metadata、tensor index 和 payload offset | SafeTensors、分片 checkpoint、远程下载 | 先固定一种真实格式 |
| TinyLlama 配置和 tensor 名映射 | 多模型架构注册表 | 先验证 Llama 主线 |
| CPU 权重对象和请求 Context | GPU placement、tensor parallel、pipeline parallel | 后端与分布式不属于加载入门闭环 |
| 明确失败的 shape/dtype/metadata 校验 | lazy loading、mmap 策略和权重缓存 | 先保证错误可观察 |

## 源码阅读问题

1. tensor 名称在什么时候从文件 schema 转换为模型层参数？
2. 模型权重与请求级 KV/position 状态分别由谁持有？
3. 增加一种 checkpoint 格式时，哪些内部对象和校验不应改变？
