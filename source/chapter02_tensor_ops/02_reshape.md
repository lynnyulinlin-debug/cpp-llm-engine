# 02.2 Reshape 重塑


## 用 Reshape 改变 Tensor 视图

实现只改变形状、不复制数据的 `reshape` 操作。

读完后应能做到：

- 理解 `reshape` 的语义边界
- 理解为什么 `reshape` 不能偷偷拷贝数据
- 理解 `reshape` 对后续算子的意义

## 代码直达

- <a href="../_code/chapter02_tensor_ops/src/tensor.cpp"><code>tensor.cpp</code></a>
- <a href="../_code/chapter02_tensor_ops/tests/chapter02_tensor_ops_test.cpp"><code>chapter02_tensor_ops_test.cpp</code></a>

## 核心语义

`reshape` 是最典型的 view 操作之一。它只改变张量的形状解释，不改变底层存储内容。

语义要求：

- 只改变 `shape`
- 不复制底层数据
- 保持数据总元素数一致
- 保持底层存储句柄不变

`reshape` 的关键约束有两个：元素总数必须不变，当前 view 必须逻辑连续。如果新旧 shape 的元素数不同，或者 view 中间存在行间隙，这个操作就应该直接拒绝，而不是偷偷复制数据。

本章的实现允许带非零 `offset` 的连续 view 继续 reshape。例如，先取出完整的一行，再把这一行 reshape 成 `3x1`，仍然复用同一份 storage。一个跨行的 `2x2` slice 具有父 Tensor 的行 stride，虽然共享 storage，但不是连续 view，因此不能 reshape。

最小实现通常只做四件事：

1. 校验当前 view 是否连续
2. 校验新 shape 的元素数
3. 复用原始存储和偏移
4. 返回一个只更新 shape 和 stride 的新视图

便捷入口和工程入口复用同一条核心路径：

```cpp
Tensor output;
auto status = tensor.try_reshape(rows, cols, &output);
```

`try_reshape()` 用 `Status` 区分 source 未就绪、view 非连续、元素数变化和溢出；`reshape()` 在失败时仍返回无效 Tensor。两种入口不会分配新 storage，也不会形成两套 reshape 语义。

实现建议：

- 检查元素数量是否匹配
- 复用原始存储
- 返回新的张量视图对象

## `reshape` 在真实项目里为什么重要

`reshape` 在推理里最重要的价值，不是把二维改成一维，而是让同一块数据在不同计算阶段以不同形状被解释。

学习者读真实项目时，应该先把这件事和“生成新张量”分开：

- 如果只是为了匹配下一步算子输入 shape，优先想到 `reshape` / view
- 如果真的改变了数据排布或内容，那才应该考虑新分配或复制

这也是后面看 QKV 组织、中间结果整理或某些 attention 输入变换时最需要带着的判断。

## 验证与排查

验证时重点看：

- `reshape` 前后元素总数一致
- 数据内容不变
- 新旧张量共享同一份存储
- 非连续 slice 的 reshape 会失败

一个简单的验证方式是把 `{2, 2}` reshape 成 `{4}`，再把结果转回 vector 比较，确认内容顺序没有变化。

常见问题：

- reshape 后偷偷分配了新内存
- shape 修改后元素数量不一致
- 忘记保留原始 storage
