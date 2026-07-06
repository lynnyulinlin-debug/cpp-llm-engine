# 03.7 Chapter Wrap-up 本章解决了什么

这章回答的是：矩阵乘法是否已经形成可复用、可验证、可加速的算子？

如果实现正确，`MatMul` 已经成为一个独立算子，既有可读的 CPU 基线，也有可用的 cuBLAS 加速实现。

## 1. 本章解决了什么

- `MatMul` 已经从普通运算变成稳定的算子边界
- CPU reference 负责把语义和布局说清楚
- cuBLAS 负责把性能提上来，但不会改写接口契约
- PyTorch 对照负责确认数值行为没有偏离预期

## 2. 为什么它重要

这一章真正要证明的不是“矩阵乘法能不能算”，而是“矩阵乘法能不能被封装成一个稳定边界，供 attention 和 FFN 反复复用”。

如果没有这层边界，后续结构层的问题会和算子层的问题混在一起，排错成本会迅速上升。

## 3. 背景回顾

`chapter01` 提供了权重读取，`chapter02` 提供了 view-aware Tensor。  
这一章把它们收束成一个稳定的数值算子边界，让上层只关心语义，不用直接面对裸数组或后端差异。

## 4. 中间推导回顾

从问题往下推，设计会自然收敛成这条路径：

- 先统一算子语义
- 再给出 CPU 参考实现
- 再接入高性能 backend
- 最后用 PyTorch 对照验证数值行为

这里的关键转折点是：`MatMul` 不是直接等于 `GEMM`，前者是接口语义，后者是高性能实现方式。

朴素循环作为正确性基线，cuBLAS 作为性能基线，这两条线并存，才能把正确性验证和性能优化之间的工程边界分开。`03.6` 解释了为什么这种“双基线”策略是合理的。

## 5. 你现在应该能做什么

- 解释为什么 `MatMulOp` 必须先于优化实现存在
- 说明 CPU reference 为什么是最稳的 correctness baseline
- 说明 cuBLAS 为什么可以作为后端而不是上层接口
- 通过对照测试检查布局和数值是否一致

## 6. 边界说明

这一章不负责替你证明所有性能细节，也不负责替你实现更多 backend 变体。

它只负责把语义边界和实现边界分开：

- 接口稳定
- 布局明确
- 参考路径可读
- 高性能路径可替换

## 7. 下一步是什么

进入 `chapter04_attention`，把 `MatMul` 用到多头注意力和前馈网络中，开始组装真正的 decoder 结构。

如果你对为什么参考实现要先写朴素循环还不够踏实，可以先补读 `06_complexity_intuition.md`，它专门解释这个设计选择。

## 8. 回看源码

- 主实现：[`chapter03_matmul 代码入口`](/code/chapter03_matmul)
- 正确性基线：[`chapter03_matmul 代码入口`](/code/chapter03_matmul)
- 验证入口：[`chapter03_matmul 代码入口`](/code/chapter03_matmul)

## 8. 回看源码

- 主实现：[`operator.cpp`](/code/chapter03_matmul)、[`matmul_op.cpp`](/code/chapter03_matmul)
- 正确性基线：[`matrix.cpp`](/code/chapter03_matmul)
- 验证入口：[`chapter03_matmul_test.cpp`](/code/chapter03_matmul)

## 9. 收口结论

这一章的收口结论是：`MatMul` 已经从普通运算变成可复用、可验证、可加速的算子层边界。  
下一章不再重新定义矩阵乘法，而是在这个边界上把 attention、FFN 和 residual 组合成真正的 decoder 结构。
