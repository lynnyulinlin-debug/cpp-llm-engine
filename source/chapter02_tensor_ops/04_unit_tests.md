# 02.4 Unit Tests 单元测试

## 用单元测试锁定 View 语义

用简单单元测试验证 `reshape` 和 `slice` 的 view 语义是否正确。

读完后应能做到：

- 理解为什么 view 操作必须测试
- 理解如何用最小案例验证共享存储
- 理解测试应该覆盖语义而不只是输出值

## 代码直达

- <a href="../_code/chapter02_tensor_ops/tests/chapter02_tensor_ops_test.cpp"><code>chapter02_tensor_ops_test.cpp</code></a>
- <a href="../_code/chapter02_tensor_ops/src/tensor.cpp"><code>tensor.cpp</code></a>

## 测试设计

这一章的正确性不在于“能不能跑”，而在于“有没有保持 view 语义”。如果测试缺失，后面很容易在不知不觉中引入数据复制或偏移错误。

测试重点：

- `reshape` 不复制数据
- `slice` 返回共享存储
- 切片内容和预期一致
- offset 和 row/column stride 符合预期
- 非连续 view 的 reshape 会被拒绝
- 越界访问和非法切片会失败
- 父 Tensor 销毁后，仍被持有的 view 保持有效
- core contract 能验证连续 Tensor 和带 offset slice 的最大可达存储位置
- `try_reshape()` / `try_slice()` 返回可断言的错误消息

建议测试方式：

- 固定输入
- 小尺寸张量
- 明确预期结果
- 对共享地址或偏移进行检查

如果测试环境允许，最好同时检查两个层面：

1. 数值层面：切片后的数据内容是否正确
2. 结构层面：新旧张量是否共享同一份 storage，偏移是否如预期变化

## 验证与排查

验证时重点看：

- 测试能稳定通过
- 错误实现会在测试中失败
- 视图语义被明确约束住

这类测试的意义不是覆盖所有情况，而是把最容易被误写的语义锁住，避免后续重构时把共享存储悄悄改成拷贝。

本章测试直接使用 C++ `assert`，从仓库根目录运行：

```cpp
bash code/chapter02_tensor_ops/scripts/run.sh
```

脚本会隔离构建 demo，执行 CTest，并在测试中同时验证“数据内容”“共享关系”“stride”“边界”和“生命周期”。

常见问题：

- 只测数值，不测共享关系
- 只测正向路径，不测边界情况
- 用太大的输入掩盖了语义错误
