# 06.4 DequantizeOp 反量化算子

对应代码：见 [本章代码索引](/code/chapter06_quantization)

## 这一页要解决什么

实现 `DequantizeOp`，把 INT8 数据反量化回 FP16 或同类浮点表示，供后续计算使用。

## 读完你应该能做什么

- 理解反量化的作用
- 理解 scale 和 dtype 转换的关系
- 理解反量化在推理链路中的位置

## 为什么这一步重要

量化权重不一定能直接参与所有算子计算。反量化是把压缩表示重新变成可计算表示的重要桥梁。

## 这一页的核心内容

- INT8 -> FP16
- scale 应用
- 反量化输出张量

`DequantizeOp` 是量化和计算之间的桥梁。它的实现要尽量单一明确：把量化输入变成可用于后续算子的浮点输出，不要顺手做别的优化，以免调试困难。

如果后续出现融合需求，也应该在这个明确的语义之上扩展，而不是一开始就把逻辑写散。

本教程的最小实现采用全局标量 `scale`（对称量化）。对于更复杂的量化格式，比如每通道或分组量化，`scale` 应该作为 `Tensor` 传入，与 `q` 的形状或分组结构对齐，并在逐元素反量化时正确索引。接口可以先保持为 `dequantize(q, scale)`，只是 `scale` 的类型从 `float` 扩展为 `Tensor`。

源码锚点：`code/chapter06_quantization/src/dequantize_op.cpp`

把它写成独立算子的好处是，你可以单独验证“反量化公式是否正确”，然后再把它接到 Linear 或其他算子上。这样一旦误差不对，你不会同时怀疑好几层逻辑。

最小反量化伪代码可以写成：

```cpp
Tensor dequantize(const Tensor& q, float scale) {
    Tensor out(q.name(), q.shape(), DType::Float32);
    for (std::size_t i = 0; i < q.numel(); ++i) {
        out[i] = static_cast<float>(q[i]) * scale;
    }
    return out;
}
```

## 实现建议

- 先支持最基础的 INT8 到浮点转换
- 保持接口和后续 Linear 可直接连接
- 不要把反量化逻辑散落在多个算子里

## 怎么验证

- 反量化结果和预期接近
- 输出 dtype 正确
- 数值误差在可接受范围内

最小验证可以直接选择几个整数样本，手动算出对应的浮点结果，然后和 op 输出对照。

## 常见问题

- scale 使用错误
- 输出 dtype 选择不一致
- 把反量化逻辑塞进别的算子里
