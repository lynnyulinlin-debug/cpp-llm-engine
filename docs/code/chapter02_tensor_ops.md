# chapter02_tensor_ops 代码导航

本章代码包含共享 Storage 的 Tensor view、reshape、slice、示例和测试。

## 阅读顺序

1. 先看 `tensor.h`
2. 再看 `tensor.cpp`
3. 最后看示例和测试

## 视图语义

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter02_tensor_ops/include/chapter02_tensor_ops/tensor.h"><code>tensor.h</code></a> | Tensor、`contract()` 和 `try_*` 接口 |
| <a href="/_code/chapter02_tensor_ops/src/tensor.cpp"><code>src/tensor.cpp</code></a> | reshape / slice 与 core contract 适配 |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter02_tensor_ops/examples/main_tensor_ops.cpp"><code>examples/main_tensor_ops.cpp</code></a> | 示例入口 |
| <a href="/_code/chapter02_tensor_ops/tests/chapter02_tensor_ops_test.cpp"><code>tests/chapter02_tensor_ops_test.cpp</code></a> | 章节测试 |
| <a href="/_code/chapter02_tensor_ops/scripts/run.sh"><code>scripts/run.sh</code></a> | 运行脚本 |

## 对应文档

- [chapter02 导读](../chapter02_tensor_ops/intro.md)
