# 02.3 Slice 切片


## 这一页要解决什么

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
- 记录偏移和长度
- 不复制原始数据

`slice` 和 `reshape` 的区别在于：`reshape` 改的是“如何解释整个数据”，`slice` 改的是“读哪一段数据”。因此 `slice` 必须记录起始偏移，并把视图限制在目标区间内。

最小版本可以先支持一维或二维切片，先把偏移和长度算对，再考虑更复杂的多维索引。

示例伪代码可以写成：

```cpp
Tensor slice(const Tensor& t, std::size_t begin, std::size_t len) {
    Tensor out = t;
    out.set_offset_bytes(t.offset_bytes() + begin * dtype_size(t.dtype()));
    out.set_shape({static_cast<int64_t>(len)});
    return out;
}
```

如果是二维切片，偏移计算会更复杂，但思想还是一样：共享 storage，只改视图边界。

实现建议：

- 计算切片起点对应的偏移
- 保留原始 storage
- 更新视图范围和 shape

## `slice` 在真实项目里为什么重要

`slice` 这一页和推理引擎的联系，比看起来更直接。很多真实工程问题本质上都不是“做一次新计算”，而是“从已有数据里取出该看的那一段”。

最典型的对应场景包括：

- token 范围的局部读取
- attention 中间结果的局部窗口
- KV cache block 或 chunk 的局部访问

所以这一页真正要建立的不是切片语法，而是一种工程习惯：先问能不能通过 view 取局部窗口，再问要不要复制出一份新数据。

## 验证与排查

验证时重点看：

- 子视图能读到原始数据中的对应片段
- 父视图和子视图共享同一存储
- 视图销毁不会破坏底层数据

建议用一个小数组做测试，例如 `{10, 20, 30, 40, 50}`，切出中间一段，确认返回内容和原始数组对应区间一致。

常见问题：

- 切片偏移计算错误
- 返回了数据拷贝而不是视图
- shape 与实际窗口不一致
