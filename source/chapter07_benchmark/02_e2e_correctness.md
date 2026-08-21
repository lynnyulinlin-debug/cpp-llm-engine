# 07.2 End-to-End Correctness Testing 端到端正确性测试


## 这一页要解决什么

通过端到端测试，把当前推理结果和 PyTorch golden logits 对照起来，确认整条链路正确。

读完后应能做到：

- 理解为什么需要端到端正确性测试
- 理解 golden logits 的作用
- 理解 cosine similarity 的比较方式

## 阅读入口

- 前置阅读：[`07.1 Test Harness Basics`](./01_test_harness_setup.md)、[`chapter05_generation`](../chapter05_generation/intro.md)
- 源码直达：<a href="../_code/chapter07_benchmark/tests/chapter07_benchmark_test.cpp"><code>chapter07_benchmark_test.cpp</code></a>、<a href="../_code/chapter07_benchmark/src/test_helpers.cpp"><code>test_helpers.cpp</code></a>
- 相关阅读：[`07.3 Benchmark Metrics`](./03_benchmark_metrics.md)、[`回归测试`](../reference/regression_testing.md)

## 测试设计

单个模块正确不代表整条链路正确。端到端测试是确认“从输入到输出”都没问题的最直接方式。

这一页的核心内容：

- 输入 prompt
- 跑完整推理链路
- 对照 PyTorch golden logits
- 使用 cosine similarity 评价结果

端到端正确性测试的作用是把前面章节各自正确的模块串起来，验证整体行为是否仍然正确。这里建议固定模型版本和输入，保证 golden logits 可比。

如果测试输入变了，golden 结果也要同步更新，否则对照就失去意义。

这一页可以把 golden logits 的角色写得更明确：它不是要求数值逐位完全一致，而是要求与参考实现保持足够接近。这样读者就不会把端到端测试误解成一个“必须绝对相等”的测试。

`cosine_similarity` 适合捕捉输出向量的整体方向一致性；如果 logits 有系统性缩放偏置，再补 `mean_absolute_error` 或 `mse` 来看量级。两者一起用通常比单一指标更可靠。

端到端对照伪代码可以写成：

```cpp
auto golden = pytorch_forward(prompt);
auto actual = engine_forward(prompt);
EXPECT_GT(cosine_similarity(golden, actual), 0.99f);
```

## 验证与排查

验证标准：

- cosine similarity 达到预期阈值
- 输出形状正确
- 结果稳定可复现

如果相同输入在多次运行中都落在同一误差范围内，说明这条链路已经足够稳定，可以作为回归基线。

常见问题：

- golden logits 不匹配当前模型版本
- 输入预处理不一致
- 指标阈值和场景不匹配
