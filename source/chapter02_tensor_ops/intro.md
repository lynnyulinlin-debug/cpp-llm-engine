# 02.0 Tensor Views and Shared Storage 学习视图语义和存储共享

这一章先回答一个视图问题：张量怎样在不拷贝数据的情况下改变形状和局部窗口？

本章会为 `Tensor` 增加 view 语义的基础操作：`reshape` 和 `slice`。

## 本章和 vLLM / nano-vLLM 的联系

真实推理框架里很多张量变化不是复制数据，而是改变 shape、offset、view 或 metadata。vLLM / nano-vLLM 的模型输入、attention 中间结果和 KV cache 视图都会依赖这种“不移动数据，只改变观察方式”的思路；本章先把这层 Tensor view 语义讲清楚。

更具体地说，本章处在真实推理执行链里一个很容易被忽略的位置：

- 上层模型或 runtime 需要不同形状、不同窗口的数据观察方式
- Tensor 层负责把这些变化收束成 view，而不是新的数据块
- 后面的 attention、KV cache 和 decode 路径才能在不额外复制的前提下继续往下走

学习本章时，最重要的不是把 `reshape`、`slice` 当成通用 API，而是先建立一个判断：这次变化是在“重新分配数据”，还是只是在“重新观察已有数据”。

## 本章概览

| 先看什么 | 这一节讲什么 | 为什么重要 | 对应小节 |
| --- | --- | --- | --- |
| 共享存储为什么重要 | `shared_ptr` 和底层 storage | 先让多个 view 能共享数据 | [`02.1 Shared Underlying Storage`](./01_shared_ptr_memory.md) |
| `reshape` 做什么 | 只改 shape，不复制数据 | 先分清形状解释和数据本体 | [`02.2 Reshape`](./02_reshape.md) |
| `slice` 做什么 | 窗口、offset 和子视图 | 先固定局部观察语义 | [`02.3 Slice`](./03_slice.md) |
| 怎么验证 view | 单元测试和边界检查 | 先锁住共享存储语义 | [`02.4 Unit Tests`](./04_uint_tests.md) |
| 本章怎么收口 | view-aware Tensor 和下一章入口 | 进入算子前先固定张量模型 | [`02.5 本章小结`](./summary.md) |

## 阅读入口

- 前置阅读：[`chapter01_tensor_loader`](../chapter01_tensor_loader/intro.md)、[`foundation`](../foundation/README.md)
- 源码直达：<a href="../_code/chapter02_tensor_ops/include/chapter02_tensor_ops/tensor.h"><code>tensor.h</code></a>、<a href="../_code/chapter02_tensor_ops/src/tensor.cpp"><code>tensor.cpp</code></a>、<a href="../_code/chapter02_tensor_ops/tests/chapter02_tensor_ops_test.cpp"><code>chapter02_tensor_ops_test.cpp</code></a>
- 相关阅读：[`02.1 Shared Underlying Storage`](./01_shared_ptr_memory.md)、[`02.4 Uint Tests`](./04_uint_tests.md)、[`推理引擎内存管理`](../reference/memory_management.md)、[`02.5 本章小结`](./summary.md)

## 这章在解决什么

这一章只做一件事：让视图改变不再依赖拷贝。它要把 `Tensor` 从“数据容器”推进到“共享存储的视图对象”，让多个张量对象共享同一份底层存储，并在不复制数据的情况下改变观察方式。

它解决的不是“少写一次 memcpy”这么局部的问题，而是后面真实推理路径里为什么能够同时做到：

- 改 shape
- 取局部窗口
- 共享 KV cache 或中间结果

而不把内存成本和数据移动成本一路放大。

## 这章怎么做

最小设计只完成两件事：

- 共享底层存储
- 让 `reshape` 和 `slice` 只改变观察方式，不复制数据

推理引擎里很多变换只是在改变观察方式，不是在生成新数据块。把这些约束摆出来后，设计就会自然收敛成“底层 storage + view 对象”。

## 收口与下一步

这一章留下的判定标准很简单：以后看到张量操作，先问它是在改“形状解释”，还是在改“观察窗口”。如果想验证得更完整，可以检查：

- `reshape` 只改变 shape，不复制底层数据
- `slice` 只改变窗口和 offset，不切断 storage
- 父张量和子视图共享同一份底层存储

下一步进入 `chapter03_matmul`，开始把 `Tensor` 用到真正的数值计算中。`chapter02` 把“同一份数据怎么被看”说清楚了，`chapter03` 才开始回答“这份数据怎么参与计算”。
