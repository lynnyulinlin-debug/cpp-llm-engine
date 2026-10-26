# Chapter 02：Tensor Ops Tensor 视图与共享存储

本目录实现不复制底层 storage 的 `reshape`、`slice` 和 view 语义，为后续算子提供稳定 Tensor 行为。完整学习说明见 [`source/chapter02_tensor_ops`](../../source/chapter02_tensor_ops/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `CMakeLists.txt` | library、demo 和 test target |
| `src/tensor.cpp` | reshape、slice、共享 storage 和 core contract 适配 |
| `examples/main_tensor_ops.cpp` | 最小 view 操作演示 |
| `tests/chapter02_tensor_ops_test.cpp` | shape、offset、别名和边界测试 |
| `scripts/run.sh` | 隔离构建、CTest 和 demo 入口 |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter02_tensor_ops/scripts/run.sh
```

重点观察：view 修改是否反映到原始 Tensor，`contract()` 是否保留 shape/stride/offset，`try_reshape()` 和 `try_slice()` 是否为非法输入返回明确 `Status`。原 `reshape()` / `slice()` 仍保留适合入门的无效 Tensor 返回方式。
