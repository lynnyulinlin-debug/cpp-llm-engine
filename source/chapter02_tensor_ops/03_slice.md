# 02.3 Slice 切片


## 用 Slice 取局部视图

实现返回子视图的 `slice` 操作，让张量可以表达局部窗口而不复制数据。

读完后应能做到：

- 理解 `slice` 和 `reshape` 的区别
- 理解子视图为什么要带偏移
- 理解如何在共享存储上构造局部窗口

## 代码直达

- <a href="../_code/chapter02_tensor_ops/src/tensor.cpp"><code>tensor.cpp</code></a>
- <a href="../_code/chapter02_tensor_ops/tests/chapter02_tensor_ops_test.cpp"><code>chapter02_tensor_ops_test.cpp</code></a>

## 核心语义

很多后续场景都需要局部访问张量，例如分块计算、子矩阵访问、token 范围处理。`slice` 是这些能力的基础。

语义要求：

- 返回子视图
- 共享底层数据
- 记录偏移、shape 和 stride
- 不复制原始数据

`slice` 和 `reshape` 的区别在于：`reshape` 改的是“如何解释整个数据”，`slice` 改的是“读哪一段数据”。因此 `slice` 必须记录起始偏移，并把视图限制在目标区间内。

最小版本先支持二维切片，先把偏移、shape 和 stride 算对，再考虑更复杂的多维索引。

本章二维 slice 的核心计算是：

```cpp
const std::size_t new_offset = offset_ + row_start * row_stride_ + col_start * col_stride_;
return Tensor(storage_, row_count, col_count, new_offset, row_stride_, col_stride_);
```

二维切片的偏移是父视图偏移、起始行 stride 和起始列 stride 的组合；返回 view 还要保留父视图的 stride。这样可以正确表达跨行窗口，但这类窗口通常不是连续 view，不能直接 reshape。

`try_slice()` 将“起点越界”“长度为零”“范围超出 shape”和“offset 溢出”分别返回为 `Status`。`contract()` 再把结果导出为 `{shape, strides, storage_offset_elements, storage_bytes}`，后续算子无需了解 `shared_ptr<vector<float>>` 也能检查 view 范围。

实现建议：

- 计算切片起点对应的偏移
- 保留原始 storage
- 更新视图范围和 shape

## `slice` 在真实项目里为什么重要

`slice` 和推理引擎的联系很直接。很多真实工程问题本质上是“从已有数据里取出该看的那一段”。

最典型的对应场景包括：

- token 范围的局部读取
- attention 中间结果的局部窗口
- KV cache block 或 chunk 的局部访问

切片练习要建立一种工程习惯：先问能不能通过 view 取局部窗口，再问要不要复制出一份新数据。

## 验证与排查

验证时重点看：

- 子视图能读到原始数据中的对应片段
- 父视图和子视图共享同一存储
- 父 Tensor 销毁后，仍被持有的 view 可以继续读写
- 越界起点、长度和元素访问会失败

建议用一个小数组做测试，例如 `{10, 20, 30, 40, 50}`，切出中间一段，确认返回内容和原始数组对应区间一致。

常见问题：

- 切片偏移计算错误
- 返回了数据拷贝而不是视图
- shape 与实际窗口不一致
