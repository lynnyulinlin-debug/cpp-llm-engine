# 06.2 Extend Tensor 扩展 Tensor


对应代码：见 [本章代码索引](/code/chapter06_quantization)

## 这一页要解决什么

让 `Tensor` 支持 INT8 dtype，为量化权重和量化计算打基础。

## 读完你应该能做什么

- 理解为什么 Tensor 需要支持更多 dtype
- 理解 INT8 张量在存储层面的变化
- 理解扩展 dtype 后对后续算子的影响

## 为什么这一步重要

量化不是单独存在的。它必须落到张量表达上，才能让模型加载、算子和验证全部走通。

## 这一页的核心内容

- 在 `Tensor` 中加入 INT8 dtype
- 保留元信息和存储表示
- 让后续加载器和算子能识别量化张量

扩展 dtype 时，最好把“元素类型”和“底层存储”区分开看。dtype 告诉系统如何解释数据，storage 则保存实际字节。这样后面无论是 FP32、INT8 还是其他表示，都可以沿用同一套 Tensor 框架。

这也意味着你不应该把量化逻辑直接塞进 Tensor 的行为里。Tensor 负责表达数据，量化和反量化应该留给专门的算子或加载逻辑处理，这样边界才清楚。

在实现中，Tensor 可以持有 `std::vector<uint8_t>` 作为底层存储，而 `dtype` 决定从这个字节数组里按什么宽度、什么方式读取元素。例如 `dtype == DType::Float32` 时，从字节偏移 `i * 4` 读取 `float`；`dtype == DType::Int8` 时，从偏移 `i` 读取 `int8_t`。这种设计使新增 dtype 时通常只需要扩展读取逻辑，而不需要改动存储层。

最小 dtype 扩展伪代码可以写成：

```cpp
enum class DType : uint8_t {
    Float32,
    Int8,
};
```

再由 `dtype_size()` 和 `deduce_dtype()` 去完成类型和字节宽度的映射。

## 实现建议

- 保持 dtype 枚举可扩展
- 不破坏原有 FP16 / FP32 路径
- 让量化 dtype 和 storage 表示解耦

## 怎么验证

- Tensor 可以表达 INT8 数据
- 旧的浮点路径不受影响
- dtype 选择可被正确区分

如果一个旧的 FP32 测试在扩展 INT8 后仍然通过，说明这次扩展没有破坏原有主线。

## 常见问题

- 只加 dtype，没处理存储表示
- 破坏了原有非量化路径
- 量化 dtype 设计过于死板
