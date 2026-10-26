# Chapter 03：MatMul 与计算后端

本目录把 CPU reference、算子接口和 backend 分派连接起来，再为 cuBLAS 保留扩展边界。完整学习说明见 [`source/chapter03_matmul`](../../source/chapter03_matmul/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `include/` | `Operator`、`MatMulOp`、backend 和请求契约 |
| `src/operator.cpp` | 算子公共行为 |
| `src/matmul_op.cpp` | MatMul 算子、`try_compute` 和 backend 请求适配 |
| `src/cpu_backend.cpp` | CPU reference backend 分派 |
| `src/cpu_kernel.cpp` | row-major CPU 三重循环 kernel |
| `src/matrix.cpp` | 教学矩阵存储与计算 |
| `tests/chapter03_matmul_test.cpp` | 数值正确性、shape 和边界检查 |
| `tests/operator_smoke_test.cpp` | Operator/backend 接口 smoke |
| `examples/main_matmul.cpp` | demo 输出和调用示例 |
| `scripts/run.sh` | 隔离构建、CTest 和 demo 入口 |
| `code/core` | 共享 `Backend`、`Operator`、`Status` 契约 |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter03_matmul/scripts/run.sh
```

`Matrix::contract()` 把连续二维存储适配为公共 Tensor 元数据；`try_compute()` 用 `Status` 贯通 Operator 与 backend，原 `compute()` 继续提供异常式兼容入口。GPU backend 或 cuBLAS 必须复用同一输入输出约定和数值结果。
