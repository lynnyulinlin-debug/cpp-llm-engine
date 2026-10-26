# 06.3 Load Quantized Weights 加载量化权重


## 把量化权重读入程序

修改 `ModelLoader`，让它可以读取量化权重文件或 GGUF 中的量化数据。

读完后应能做到：

- 理解量化权重在文件中的组织方式
- 理解 loader 如何识别量化 dtype
- 理解量化数据如何进入程序内部

**阅读入口**

- 前置阅读：[`06.1 Quantization Basics`](./01_quantization_basics.md)、[`06.2 Extend Tensor`](./02_extend_tensor.md)
- 源码直达：<a href="../_code/chapter06_quantization/src/tensor.cpp"><code>tensor.cpp</code></a>、<a href="../_code/chapter06_quantization/src/quantized_linear.cpp"><code>quantized_linear.cpp</code></a>、<a href="../_code/chapter06_quantization/tests/chapter06_quantization_test.cpp"><code>chapter06_quantization_test.cpp</code></a>
- 相关阅读：[`06.4 DequantizeOp`](./04_dequantize_op.md)、[`06.5 Quantized Linear`](./05_linear_with_quant.md)

## 核心设计

模型只有在加载环节真正读懂量化格式，后面的反量化和量化 Linear 才有意义。

量化权重加载需要覆盖：

- 量化权重文件解析
- 量化 dtype 识别
- 量化 tensor 的构建

loader 这里最重要的是“区分元信息和数据”。文件里不仅有字节，还要有 dtype、shape、scale 之类的信息，loader 的职责就是把这些信息搬进内存对象。

最小版本可以先支持一种量化格式，后续再扩展到更多变体。

如果量化格式还带有分组大小、block size 或零点等参数，也要明确保存，否则后面 dequantize 时就只剩下原始字节，没有足够上下文还原数值。

一个最小化的 `QuantMeta` 可以包含：

- `dtype`，例如 `QInt8`
- `shape`
- `scale`，可以是标量，也可以是按通道数组
- `zero_point`，可选

如果是分组量化，还应该额外保存：

- `block_size`
- 每个分组对应的 `scales`
- 每个分组对应的 `zero_points`

这些信息在加载时必须完整保留，因为反量化要依赖它们还原数值。

最小 loader 伪代码可以写成：

```cpp
Tensor load_quant_tensor(std::istream& in) {
    Meta meta = read_meta(in);
    std::vector<std::byte> raw = read_bytes(in, meta.byte_size);
    return Tensor::from_raw_bytes(meta.name, meta.shape, meta.dtype, std::move(raw));
}
```

真正复杂的地方是 meta 里要包含足够的信息支撑后续反量化。

实现建议：

- 在现有 loader 上扩展，而不是重写全部逻辑
- 保持量化和非量化路径并存
- 清楚地区分 metadata 和 raw data

## 验证与排查

验证时重点看：

- 量化权重可以被正确读取
- tensor 数量和名称合理
- dtype 信息能正确传递

如果加载后的 quant tensor 可以和原始权重在元信息上对得上，这一步通常就已经可靠了。

常见问题：

- 量化格式识别失败
- dtype 和数据内容不一致
- loader 为量化路径引入副作用
