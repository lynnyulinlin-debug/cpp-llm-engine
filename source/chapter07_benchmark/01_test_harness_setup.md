# 07.1 Test Harness Basics 测试 harness 基础


## 先固定测试入口

引入一个统一的测试 harness，并用它来编写算子和模块的单元测试。

读完后应能做到：

- 理解为什么需要统一的测试入口
- 理解如何为算子和模块建立测试入口
- 理解测试代码和业务代码的边界

**阅读入口**

- 前置阅读：[`chapter07 导读`](./intro.md)、[`chapter05_generation`](../chapter05_generation/intro.md)
- 源码直达：<a href="../_code/chapter07_benchmark/src/test_helpers.cpp"><code>test_helpers.cpp</code></a>、<a href="../_code/chapter07_benchmark/tests/chapter07_benchmark_test.cpp"><code>chapter07_benchmark_test.cpp</code></a>
- 相关阅读：[`07.2 End-to-End Correctness Testing`](./02_e2e_correctness.md)、[`07.3 Benchmark Metrics`](./03_benchmark_metrics.md)

## 测试设计

单元测试是整个教程可维护性的基础。如果没有统一测试框架，后面每个章节的验证方式都会不一致。

测试入口需要覆盖：

- 统一测试入口
- 命名测试用例
- 算子单元测试

测试入口的第一目标是稳定和可复现。因此这里先接起最小测试入口，再逐步增加测试组织方式。

如果后面每章都能沿用同样的测试风格，整个教程的维护成本会低很多。

测试 harness 的文件结构建议和源码结构对齐：`tests/` 目录下为每个算子或模块建立对应的测试文件，比如 `test_matmul.cpp`、`test_attention.cpp`、`test_decoder_layer.cpp`。每个测试文件可以包含多个测试用例，用例命名遵循 `test_<模块>_<场景>` 的格式，比如 `test_matmul_small_shape`、`test_matmul_error_shape`。这种组织方式可以让测试和源码的映射关系一目了然。

测试入口还要遵守一个约束：单测应该尽量只依赖局部构造的数据，而不是依赖外部大模型文件。这样测试才快，也才更适合做回归。

最小测试结构可以写成：

```cpp
bool test_small_shape() {
    Tensor a = make_tensor({2, 3});
    Tensor b = make_tensor({3, 4});
    Tensor c = matmul(a, b);
    return c.shape() == std::vector<int64_t>({2, 4});
}
```

实现建议：

- 先从最简单的测试开始
- 保持测试样例小而明确
- 让失败信息足够清晰

## 验证与排查

验证时重点看：

- 可以编写并运行测试
- 测试能够覆盖关键算子
- 测试失败时能给出明确反馈

能够在本章里快速跑通一个失败案例，并从输出中准确定位问题，说明测试框架已经具备工程价值。

常见问题：

- 测试和实现耦合过深
- 测试样例太大，不易定位问题
- 没有统一命名和组织方式
