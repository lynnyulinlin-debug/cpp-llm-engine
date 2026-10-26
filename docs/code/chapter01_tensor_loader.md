# chapter01_tensor_loader 代码导航

本章代码包含 Tensor、GGUF Reader、模型配置、Tokenizer adapter、ModelBuilder 和真实模型测试。

## 阅读顺序

1. 先看 `src/tensor.h` 和 `src/tensor.cpp`
2. 再看 `src/model_loader.h` 和 `src/model_loader.cpp`
3. 最后看测试文件

## 输入语义

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter01_tensor_loader/src/tensor.h"><code>src/tensor.h</code></a> | Tensor 接口 |
| <a href="/_code/chapter01_tensor_loader/src/tensor.cpp"><code>src/tensor.cpp</code></a> | Tensor 实现 |
| <a href="/_code/chapter01_tensor_loader/src/model_loader.h"><code>src/model_loader.h</code></a> | ModelLoader 接口 |
| <a href="/_code/chapter01_tensor_loader/src/model_loader.cpp"><code>src/model_loader.cpp</code></a> | ModelLoader 实现 |
| <a href="/_code/chapter01_tensor_loader/src/tokenizer.h"><code>src/tokenizer.h</code></a> | Tokenizer 接口和 token 类型 |
| <a href="/_code/chapter01_tensor_loader/src/tokenizer.cpp"><code>src/tokenizer.cpp</code></a> | 教学用 tokenizer 实现 |
| <a href="/_code/chapter01_tensor_loader/src/model_config.h"><code>src/model_config.h</code></a> | ModelConfig 和 metadata 校验 |
| <a href="/_code/chapter01_tensor_loader/src/model_config.cpp"><code>src/model_config.cpp</code></a> | vocab、shape 和 dtype 一致性检查 |
| <a href="/_code/chapter01_tensor_loader/src/model_metadata.h"><code>src/model_metadata.h</code></a> | 外部格式到内部 metadata 的边界 |
| <a href="/_code/chapter01_tensor_loader/src/model_metadata.cpp"><code>src/model_metadata.cpp</code></a> | metadata 基础有效性检查 |
| <a href="/_code/chapter01_tensor_loader/src/model_metadata_adapter.h"><code>src/model_metadata_adapter.h</code></a> | GGUF key/value 到 ModelMetadata 的适配接口 |
| <a href="/_code/chapter01_tensor_loader/src/model_metadata_adapter.cpp"><code>src/model_metadata_adapter.cpp</code></a> | TinyLlama/Llama 风格 metadata key 映射和错误检查 |
| <a href="/_code/chapter01_tensor_loader/src/gguf_reader.h"><code>src/gguf_reader.h</code></a> | GGUF 文件索引接口 |
| <a href="/_code/chapter01_tensor_loader/src/gguf_reader.cpp"><code>src/gguf_reader.cpp</code></a> | GGUF header、标量 metadata 和 tensor index 读取 |
| <a href="/_code/chapter01_tensor_loader/src/gguf_tensor_loader.h"><code>src/gguf_tensor_loader.h</code></a> | GGUF tensor payload loader 接口 |
| <a href="/_code/chapter01_tensor_loader/src/gguf_tensor_loader.cpp"><code>src/gguf_tensor_loader.cpp</code></a> | offset、shape、dtype 和文件边界检查 |
| <a href="/_code/chapter01_tensor_loader/src/model_builder.h"><code>src/model_builder.h</code></a> | ModelMetadata、TokenizerMetadata 和 Tensor map 的组装接口 |
| <a href="/_code/chapter01_tensor_loader/src/model_builder.cpp"><code>src/model_builder.cpp</code></a> | vocab、embedding shape 和 payload 校验 |

## 示例

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter01_tensor_loader/examples/gguf_inspect.cpp"><code>examples/gguf_inspect.cpp</code></a> | 检查真实 GGUF，并可加载第一个 Q4_K tensor |
| <a href="/_code/chapter01_tensor_loader/src/tokenizer_metadata.h"><code>src/tokenizer_metadata.h</code></a> | 真实 tokenizer metadata 契约 |
| <a href="/_code/chapter01_tensor_loader/src/tokenizer_metadata.cpp"><code>src/tokenizer_metadata.cpp</code></a> | tokenizer vocabulary、scores、token types 和 special tokens 适配 |

## 验证

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter01_tensor_loader/tests/chapter01_tensor_loader_test.cpp"><code>tests/chapter01_tensor_loader_test.cpp</code></a> | 权重加载与 Tensor 验证 |

## 对应文档

- [chapter01 导读](../chapter01_tensor_loader/intro.md)
- [输入契约与 Tokenizer](../chapter01_tensor_loader/04_input_contract_and_tokenizer.md)
- [模型元数据与 GGUF](../chapter01_tensor_loader/05_model_metadata_and_gguf.md)
