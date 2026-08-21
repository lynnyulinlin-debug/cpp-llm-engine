# 03.8 Chapter Wrap-up 本章解决了什么

这章回答的是：矩阵乘法是否已经形成可复用、可验证、可加速的算子？

如果实现正确，`MatMul` 已经成为一个独立算子，既有可读的 CPU 基线，也有可用的 cuBLAS 加速实现。

## 本章解决了什么

- `MatMul` 已经从普通运算变成稳定的算子边界
- CPU reference 负责把语义和布局说清楚
- cuBLAS 负责把性能提上来，但不会改写接口契约
- PyTorch 对照负责确认数值行为没有偏离预期
- 本章已经能对照 vLLM / nano-vLLM 里的 Linear、QKV projection、FFN 和 backend / kernel 分层

## 为什么重要

这一章把 `chapter02` 的 view-aware Tensor 收束成一个稳定的数值算子边界，让上层只关心语义，不用直接面对裸数组或后端差异。`chapter02` 解决数据怎么看，`chapter03` 解决数据怎么算。

读真实推理项目时，这一章提供的是“算子语义和后端实现分离”的判断标准：同一个 Linear 语义可以由 CPU reference、cuBLAS、自定义 CUDA kernel 或更高性能 backend 执行。

## 下一步

进入 `chapter04_attention`，把 `MatMul` 用到多头注意力和前馈网络中，开始组装真正的 decoder 结构。

## 复盘路径

- 先看 <a href="../_code/chapter03_matmul/src/operator.cpp"><code>operator.cpp</code></a> 和 <a href="../_code/chapter03_matmul/src/matmul_op.cpp"><code>matmul_op.cpp</code></a>
- 再看 <a href="../_code/chapter03_matmul/src/matrix.cpp"><code>matrix.cpp</code></a>
- 最后看 <a href="../_code/chapter03_matmul/tests/chapter03_matmul_test.cpp"><code>chapter03_matmul_test.cpp</code></a>
- 复盘整章时，回到 [`chapter03 导读`](./intro.md)
