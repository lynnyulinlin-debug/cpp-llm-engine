# 06.3 Load Quantized Weights 加载量化权重


## 这一页要解决什么

修改 `ModelLoader`，让它可以读取量化权重文件或 GGUF 中的量化数据。

## 阅读入口

- 前置阅读：[`06.1 Quantization Basics`](/chapter06_quantization/01_quantization_basics)、[`06.2 Extend Tensor`](/chapter06_quantization/02_extend_tensor)
- 源码直达：[`chapter06_quantization 代码入口`](/code/chapter06_quantization)
- 相关阅读：[`06.4 DequantizeOp`](/chapter06_quantization/04_dequantize_op)、[`06.5 Quantized Linear`](/chapter06_quantization/05_linear_with_quant)

## 代码直达

- [`src/tensor.cpp`](/code/chapter06_quantization#tensor-cpp)
- [`src/dequantize_op.cpp`](/code/chapter06_quantization#dequantize-op-cpp)
- [`src/quantized_linear.cpp`](/code/chapter06_quantization#quantized-linear-cpp)
- [`tests/chapter06_quantization_test.cpp`](/code/chapter06_quantization#chapter06-quantization-test-cpp)

## 读完后应能做什么

- 理解量化权重在文件中的组织方式
- 理解 loader 如何识别量化 dtype
- 理解量化数据如何进入程序内部

## 为什么这一步重要

模型只有在加载环节真正读懂量化格式，后面的反量化和量化 Linear 才有意义。

## 这一页的核心内容

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

## 实现建议

- 在现有 loader 上扩展，而不是重写全部逻辑
- 保持量化和非量化路径并存
- 清楚地区分 metadata 和 raw data

## 怎么验证

- 量化权重可以被正确读取
- tensor 数量和名称合理
- dtype 信息能正确传递

如果加载后的 quant tensor 可以和原始权重在元信息上对得上，这一步通常就已经可靠了。

## 常见问题

- 量化格式识别失败
- dtype 和数据内容不一致
- loader 为量化路径引入副作用
