# chapter07_benchmark 代码导航

本章代码包含 benchmark record、测试 helper、多请求 workload 和结果文件。

## 阅读顺序

1. 先看 `benchmark.cpp`
2. 再看 `test_helpers.cpp`
3. 最后看 benchmark 示例和测试

## 验证与比较

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter07_benchmark/include/chapter07_benchmark/benchmark.h"><code>benchmark.h</code></a> | benchmark 接口 |
| <a href="/_code/chapter07_benchmark/src/benchmark.cpp"><code>src/benchmark.cpp</code></a> | benchmark 实现 |
| <a href="/_code/chapter07_benchmark/include/chapter07_benchmark/test_helpers.h"><code>test_helpers.h</code></a> | 测试 helper 接口 |
| <a href="/_code/chapter07_benchmark/src/test_helpers.cpp"><code>src/test_helpers.cpp</code></a> | 测试 helper 实现 |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter07_benchmark/examples/main_benchmark.cpp"><code>examples/main_benchmark.cpp</code></a> | benchmark 示例 |
| <a href="/_code/chapter07_benchmark/tests/chapter07_benchmark_test.cpp"><code>tests/chapter07_benchmark_test.cpp</code></a> | 章节测试 |
| <a href="/_code/chapter07_benchmark/scripts/run.sh"><code>scripts/run.sh</code></a> | 运行脚本 |

## 对应文档

- [chapter07 导读](../chapter07_benchmark/intro.md)
