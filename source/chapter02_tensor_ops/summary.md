# 02.5 Chapter Wrap-up 本章解决了什么

这章回答的是：张量如何同时保留共享存储和 view 语义？

如果实现正确，`Tensor` 就不只是一个装数据的容器，而是一个可以共享底层存储、返回视图、并在不复制数据的情况下改变形状和局部窗口的对象。

## 本章解决了什么

- `Tensor` 已经能表达共享存储和 view 语义
- `reshape` 和 `slice` 已经不会默认复制数据
- 底层 storage 和 Tensor view 的职责已经分开

## 为什么重要

这一步的意义不只是省内存，更是把数据对象和视图对象分开。`chapter01` 解决输入，`chapter02` 解决视图；两章合在一起，张量前半段就稳定了。

读真实推理项目时，这一章提供的是“先判断是不是 view 问题”的标准：同一份数据可以被不同 shape、不同窗口、不同 offset 观察，而不必立刻生成新副本。很多看起来像张量处理细节的问题，实际最后都会落到 storage 共享和 view 语义上。

## 下一步

进入 `chapter03_matmul`，开始把 `Tensor` 用到真正的数值计算中。

## 复盘路径

- 先看 <a href="../_code/chapter02_tensor_ops/include/chapter02_tensor_ops/tensor.h"><code>tensor.h</code></a> 和 <a href="../_code/chapter02_tensor_ops/src/tensor.cpp"><code>tensor.cpp</code></a>
- 再看 <a href="../_code/chapter02_tensor_ops/examples/main_tensor_ops.cpp"><code>main_tensor_ops.cpp</code></a>
- 最后看 <a href="../_code/chapter02_tensor_ops/tests/chapter02_tensor_ops_test.cpp"><code>chapter02_tensor_ops_test.cpp</code></a>
- 复盘整章时，回到 [`chapter02 导读`](./intro.md)
