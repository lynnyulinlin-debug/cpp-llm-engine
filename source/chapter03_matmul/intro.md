# 03.0 MatMul and Compute Backends 构建核心矩阵乘法算子

这一章先回答一个工程问题：推理引擎里最核心的矩阵乘法应该怎样抽象和落地？

本章会把矩阵乘法封装成一个可复用的算子，先实现 CPU 参考版本，再接入 cuBLAS 作为高效实现。

## 本章和 vLLM / nano-vLLM 的联系

vLLM / nano-vLLM 里的 Linear、QKV projection、attention output projection 和 FFN 最终都会落到矩阵乘法或等价 kernel 上。本章先把 `MatMul` 做成稳定、可替换、可验证的算子边界，后面读真实项目时就能分清“上层模型语义”和“底层 backend / kernel 实现”。

更具体地说，本章处在真实推理执行链的中间层：

- 上层模型说“这里要做一次 Linear / projection / FFN 计算”
- 算子层把这件事收束成稳定的 `MatMul` 语义
- backend 层再决定交给 CPU reference、cuBLAS，还是后续更快的 kernel

学习本章时，最重要的不是记住某个库名，而是先分清：你现在看到的是“算子语义边界”，还是“backend 实现细节”。

## 本章概览

| 先看什么 | 这一节讲什么 | 为什么重要 | 对应小节 |
| --- | --- | --- | --- |
| 算子基类怎么立 | `Operator` 抽象 | 先把算子接口统一 | [`03.1 Operator Base`](./01_operator_base.md) |
| `MatMulOp` 怎么设计 | 输入输出、shape 和布局契约 | 先固定矩阵乘法语义 | [`03.2 MatMulOp Design`](./02_matmul_design.md) |
| CPU reference 做什么 | 可读正确性基线 | 先保证算对 | [`03.3 CPU Reference`](./03_cpu_reference.md) |
| cuBLAS 怎么接入 | 高性能 backend | 先保证实现可替换 | [`03.4 cuBLAS Implementation`](./04_cublas_implement.md) |
| 怎么和 PyTorch 对照 | 数值验证 | 先确认结果没有偏离 | [`03.5 Test with PyTorch`](./05_test_with_pytorch.md) |
| 为什么先写朴素实现 | 复杂度和基线直觉 | 先分清正确性和优化 | [`03.6 Complexity Intuition`](./06_complexity_intuition.md) |
| 怎么看真实 backend | Linear、QKV、FFN 和 kernel 后端 | 先建立 vLLM / nano-vLLM 阅读坐标 | [`03.7 从 MatMul 到真实推理 backend`](./07_runtime_backend_bridge.md) |
| 本章怎么收口 | 算子边界和下一章入口 | 进入结构层前先固定计算边界 | [`03.8 本章小结`](./summary.md) |

## 阅读入口

- 前置阅读：[`chapter01_tensor_loader`](../chapter01_tensor_loader/intro.md)、[`chapter02_tensor_ops`](../chapter02_tensor_ops/intro.md)
- 如果要进入 `cuBLAS` 路径，再回看 [`foundation/03_cuda_environment_basics`](../foundation/03_cuda_environment_basics.md) 和 [`foundation/07_cuda_common_libraries`](../foundation/07_cuda_common_libraries.md)
- 源码直达：<a href="../_code/chapter03_matmul/include/chapter03_matmul/operator.h"><code>operator.h</code></a>、<a href="../_code/chapter03_matmul/src/matmul_op.cpp"><code>matmul_op.cpp</code></a>、<a href="../_code/chapter03_matmul/tests/chapter03_matmul_test.cpp"><code>chapter03_matmul_test.cpp</code></a>
- 相关阅读：[`03.1 Operator Base`](./01_operator_base.md)、[`03.2 MatMulOp Design`](./02_matmul_design.md)、[`03.3 CPU Reference`](./03_cpu_reference.md)、[`03.4 cuBLAS Implementation`](./04_cublas_implement.md)、[`03.7 从 MatMul 到真实推理 backend`](./07_runtime_backend_bridge.md)

## 这章在解决什么

这一章要做的，是把矩阵乘法收束成一个稳定、可替换、可验证的算子边界：上层只依赖 `MatMulOp`，下层可以在 CPU reference、cuBLAS 或后续 backend 之间切换。

推理引擎里的大部分耗时最终都会落到矩阵乘法上，所以这章先立接口，再谈性能。它解决的不是“怎么把乘法写出来”，而是“后面 attention、FFN 和真实 backend dispatch 应该建立在哪条稳定计算边界上”。

## 这章怎么做

最小设计不是“把乘法写出来”，而是先定一条稳定的接口契约：上层只看 `MatMulOp`，下层可以切换 CPU reference、cuBLAS 或 future backend，shape、stride 和 layout contract 必须先说清楚。

`MatMul` 是“做什么”，`GEMM` 是“怎么高性能地做”。前者是算子语义，后者是实现范式。

这一层讲清楚之后，读 vLLM / nano-vLLM 时就不会把三种东西混在一起：

- 模型层在请求什么计算
- backend 层在选择哪条实现路径
- kernel 层在怎样把这条路径跑快

## 收口与下一步

这一章的验证路径很直接：先确认接口边界稳定，再确认 CPU 参考版本的结果正确，最后确认 cuBLAS 路径和 PyTorch 对照一致。

如果这一层没有先做稳，后面把 `MatMul` 嵌入 attention 和 FFN 时，定位问题会非常困难。

下一步进入 `chapter04_attention`，把这个算子边界收束进 attention、FFN 和 residual，开始组装真正的模型结构。
