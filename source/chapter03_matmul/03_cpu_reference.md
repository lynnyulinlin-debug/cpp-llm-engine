# 03.3 CPU Reference Implementation CPU 参考实现


## 先写可验证的 CPU Reference

先实现一个正确的 CPU 版本 `MatMul`，作为后续 cuBLAS 版本的对照基线。  
CPU reference 的目标是最容易读懂、最容易验证。

读完后应能做到：

- 理解为什么先做 CPU 参考实现
- 理解如何用朴素实现验证数学正确性
- 理解 correctness baseline 的意义

## 阅读入口

- 前置阅读：[`Operator Base`](./01_operator_base.md)、[`MatMulOp 设计`](./02_matmul_design.md)
- 源码直达：<a href="../_code/chapter03_matmul/src/matrix.cpp"><code>matrix.cpp</code></a>、<a href="../_code/chapter03_matmul/src/cpu_kernel.cpp"><code>cpu_kernel.cpp</code></a>、<a href="../_code/chapter03_matmul/src/matmul_op.cpp"><code>matmul_op.cpp</code></a>、<a href="../_code/chapter03_matmul/tests/chapter03_matmul_test.cpp"><code>chapter03_matmul_test.cpp</code></a>
- 相关阅读：[`cuBLAS 接入设计`](./04_cublas_implement.md)、[`PyTorch 对照测试`](./05_test_with_pytorch.md)

## 核心实现

高性能实现之前，必须先确认数学是对的。  
CPU 参考实现通常最直观，也最适合用来定位布局、转置和维度错误。

如果参考实现都不稳定，后面的 cuBLAS 路径就很难判断到底是优化引入了偏差，还是最初的数学定义就已经错了。

参考实现要尽量朴素：

- 先检查输入形状
- 再分配输出张量
- 然后用三重循环累加
- 最后返回结果

`MatMulOp::try_compute()` 负责 Matrix 与公共请求之间的适配，`CpuReferenceMatmulBackend` 只负责分派，`cpu_reference_matmul_kernel()` 校验 request 并执行三重循环。shape mismatch 不再只能依赖异常捕获，也可以直接检查返回的 `Status`。

实现时不要一开始就追求 cache 友好。  
参考实现的任务是成为正确性基线，不是成为最快版本。

最小实现里，形状检查应该直接写出来：

```cpp
assert(A.cols() == B.rows());
Tensor C(A.rows(), B.cols());
```

源码锚点：<a href="../_code/chapter03_matmul/src/cpu_kernel.cpp"><code>cpu_kernel.cpp</code></a>

这样“边界检查”和“核心计算”就会同时出现在代码里，读者也更容易把形状错误和累加错误区分开。

朴素三重循环有两个优点：

- 结构和数学定义一致，读者容易理解
- 一旦结果不对，很容易判断是维度顺序错了，还是累加逻辑错了

这也是为什么参考实现通常应该先写得“笨”一点。  
当你需要验证 cuBLAS、转置或者布局映射时，最需要的是一个可以直接对照的小实现，而不是一个高度优化但难读的版本。

## 验证与排查

验证时重点看：

- 与手工小矩阵结果一致
- 与后续 cuBLAS 结果一致
- 错误维度输入能被正确拒绝
- 空 buffer、零维度和超大 shape 能在进入循环前失败

对于小矩阵，最好直接写出手算结果。  
这样一旦数值不对，你可以立刻判断是维度顺序错了，还是累加逻辑错了。

最小伪代码可以写成：

```cpp
for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
        float sum = 0.f;
        for (std::size_t k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
```

源码锚点：<a href="../_code/chapter03_matmul/src/cpu_kernel.cpp"><code>cpu_kernel.cpp</code></a>

这段循环的价值在于可读性，而不是性能。

常见问题：

- 维度顺序搞反
- 累加顺序错误
- 为了“快”过早优化，破坏正确性
