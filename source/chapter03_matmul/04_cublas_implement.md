# 03.4 cuBLAS Backend cuBLAS 接入

本节用已验证的 `cublasSgemm` backend 说明 row-major 与 column-major 的适配。CPU reference 和 CUDA backend 复用同一个 `MatmulRequest`，GPU target 保持独立，不影响 CPU 默认回归。

读完后应能解释三个问题：项目的 MatMul 语义为什么不随 backend 改变，row-major 数据怎样交给 cuBLAS，以及 handle、stream 和 device buffer 分别由谁持有。

## 阅读入口

- 前置阅读：[`MatMulOp 设计`](./02_matmul_design.md)、[`CPU Reference`](./03_cpu_reference.md)
- 源码直达：<a href="../_code/cuda/include/cpp_llm_engine/cuda/cublas_matmul_backend.h"><code>cublas_matmul_backend.h</code></a>、<a href="../_code/cuda/src/cublas_matmul_backend.cpp"><code>cublas_matmul_backend.cpp</code></a>
- 测试直达：<a href="../_code/cuda/tests/cuda_cublas_matmul_test.cu"><code>cuda_cublas_matmul_test.cu</code></a>

## Row-major 映射

core 固定的数学语义是：

```text
C[M,N] = A[M,K] * B[K,N]
```

cuBLAS 默认按 column-major 解释内存。当前实现不复制或转置数据，而是交换传入的 A/B：`B[K,N]` 的 row-major 内存被解释为 column-major `B^T[N,K]`，`A[M,K]` 同理被解释为 `A^T[K,M]`。因此 cuBLAS 实际计算：

```text
C^T[N,M] = B^T[N,K] * A^T[K,M]
```

| SGEMM 参数 | 当前值 | 原因 |
| --- | --- | --- |
| 第一个矩阵 | `request.rhs` | 作为 `B^T[N,K]` 解释 |
| 第二个矩阵 | `request.lhs` | 作为 `A^T[K,M]` 解释 |
| `transa/transb` | `CUBLAS_OP_N / CUBLAS_OP_N` | 内存视角已经完成转置解释 |
| `m/n/k` | `N / M / K` | 输出按 `C^T[N,M]` 计算 |
| `lda/ldb/ldc` | `N / K / N` | 对应三个 column-major 矩阵的行数 |

## 资源与错误边界

`CublasMatmulBackend` 独占 move-only `cublasHandle_t`，不使用进程全局 handle。每次 `compute()` 显式接收 `CudaStream` 并调用 `cublasSetStream`，因此 handle 与执行上下文可以在下一阶段一起管理。

`CublasStatus` 保留原始 `cublasStatus_t`、操作名和可读错误，并能转换为 core `Status`。backend 在调用库函数前拒绝空 handle、无效 stream、空指针、零维和超出 cuBLAS `int` 范围的维度。

## 验证

从仓库根目录运行：

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
```

测试用 `1x1x1`、`2x3x4`、`7x5x9` 和 `31x33x17` 与 CPU 三重循环逐元素对照。固定 `256x256x256` workload 经过 1 次 warmup 后使用 event 记录 10 轮 SGEMM；D2H 和结果检查位于计时区间之外。
