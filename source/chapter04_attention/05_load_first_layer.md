# 04.5 First Layer Weight Wiring 首层权重接线

## 学习入口

Chapter01 已经建立 GGUF 索引、模型结构校验和按需加载能力。本节把这些对象接到首层计算，确认真实权重能够通过 Attention、FFN 和残差连接产生有限输出。

读完后应能做到：

- 区分教学 `CIE1 ModelLoader` 与真实 `GGUF` 加载链
- 说明首层九个权重如何从模型 schema 映射到计算模块
- 按“索引、结构、载荷、计算”四个检查点定位错误

- 前置阅读：[`04.4 DecoderLayer`](./04_decoder_layer.md)、[`Chapter01 模型加载`](../chapter01_tensor_loader/intro.md)
- 源码直达：<a href="../_code/chapter01_tensor_loader/examples/gguf_first_layer_forward.cpp"><code>gguf_first_layer_forward.cpp</code></a>、<a href="../_code/chapter01_tensor_loader/src/model_schema.cpp"><code>model_schema.cpp</code></a>、<a href="../_code/chapter01_tensor_loader/src/real_layer_weights.cpp"><code>real_layer_weights.cpp</code></a>
- 相关阅读：[`04.6 Decoder 成熟框架对照`](./06_gemma_tensorrt_comparison.md)、[`Chapter06 量化`](../chapter06_quantization/intro.md)

## 两条加载路径

Chapter01 保留两条用途不同的路径。`ModelLoader` 读取教程自定义的 CIE1 文件，适合学习二进制格式和名称索引；TinyLlama 的真实首层验证读取 GGUF，不能写成 `ModelLoader("model.bin")`。

| 路径 | 入口 | 产物 | 用途 |
| --- | --- | --- | --- |
| 教学格式 | `ModelLoader::load()` | `unordered_map<string, Tensor>` | 理解文件头、tensor 目录和 payload |
| 真实 GGUF | `GgufReader` + `ModelSchema` + `RealLlamaLayerLoader` | `RealLlamaLayerWeights` | 驱动 TinyLlama 首层和完整 Decoder |

## 四段接线

真实路径先读取索引，再把 metadata 适配为运行时配置。`ModelSchema::validate_gguf()` 检查 22 层名称和 shape，并生成稳定的 `LlamaTensorLayout`；通过结构校验后，`RealLlamaLayerLoader` 才按需读取指定层的 payload。

下面的代码摘自仓库真实调用链，省略错误输出，不是另一套示例 API：

```cpp
chapter01::GgufModelIndex index;
chapter01::GgufReader::read_index(model_path, &index, &error);

chapter01::ModelMetadata metadata;
chapter01::ModelMetadataAdapter::from_gguf(index.metadata, &metadata, &error);

chapter01::LlamaTensorLayout layout;
chapter01::ModelSchema::validate_gguf(metadata, index, &layout, &error);

chapter01::RealLlamaLayerWeights layer;
chapter01::RealLlamaLayerLoader::load(
    model_path, index, layout, 0, &layer, &error);
```

四个阶段各自只解决一个问题，因此错误可以在进入计算前被截断。

| 阶段 | 输入 | 检查结果 |
| --- | --- | --- |
| GGUF 索引 | 模型文件 | metadata、tensor 名称、类型、shape 和偏移 |
| Metadata 适配 | GGUF metadata | hidden size、层数、head 数、RoPE 与 RMSNorm 参数 |
| Schema 校验 | metadata + tensor index | Llama 所需 tensor 是否齐全且 shape 一致 |
| 首层加载 | layout + layer index 0 | 九个拥有 payload 的 `Tensor` |

## 首层权重映射

TinyLlama 使用 GGUF 的 `blk.0.*` 命名。映射由 `ModelSchema` 集中建立，计算代码不应自行拼接文件偏移或依赖 tensor 在文件中的排列顺序。

| `RealLlamaLayerWeights` 字段 | GGUF tensor | 计算位置 |
| --- | --- | --- |
| `attn_norm` | `blk.0.attn_norm.weight` | Attention 前 RMSNorm |
| `attn_q` / `attn_k` / `attn_v` | `blk.0.attn_q/k/v.weight` | Q、K、V 投影 |
| `attn_output` | `blk.0.attn_output.weight` | Attention 输出投影 |
| `ffn_norm` | `blk.0.ffn_norm.weight` | FFN 前 RMSNorm |
| `ffn_gate` / `ffn_up` | `blk.0.ffn_gate/up.weight` | SwiGLU 两条输入投影 |
| `ffn_down` | `blk.0.ffn_down.weight` | FFN 输出投影 |

## 验证顺序

首层验证按由静到动的顺序执行：先校验名称和 shape，再加载 payload，最后运行单 token 前向。真实 TinyLlama 权重包含 Q4_K、Q6_K 和浮点 norm；量化投影由 Chapter06 的 reference 或 fused 实现消费。

`gguf_first_layer_forward` 固定 token id，执行 RMSNorm、QKV 投影、GQA 单 token attention、输出投影、SwiGLU FFN 和两次残差连接。它最终检查隐藏状态长度、有限值和最大绝对值。这里的单 token attention 只验证接线，不覆盖多 token attention score、RoPE 或 KV Cache；这些能力由后续多 token 和完整 Decoder 验证承担。

从仓库根目录运行：

```bash
cmake -S code/chapter01_tensor_loader \
  -B /tmp/cpp-llm-engine-chapter01-build \
  -DCHAPTER06_SOURCE_DIR="$PWD/code/chapter06_quantization" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-engine-chapter01-build \
  --target gguf_first_layer_forward -j2
/tmp/cpp-llm-engine-chapter01-build/gguf_first_layer_forward \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

| 检查点 | 失败通常表示什么 |
| --- | --- |
| `read_index` | 文件路径、GGUF 版本或索引解析错误 |
| `from_gguf` | 必需 metadata 缺失或类型不匹配 |
| `validate_gguf` | 模型架构、层数、名称或 shape 不符合 Llama contract |
| `RealLlamaLayerLoader::load` | tensor payload 越界、类型不支持或读取失败 |
| `output_finite: true` | 首层计算已产生有限输出；不等于完整模型数值等价 |
