# 07.2 End-to-End Correctness Testing 端到端正确性测试


## 这一页要解决什么

通过端到端测试，把当前推理结果和 PyTorch golden logits 对照起来，确认整条链路正确。

## 读完你应该能做什么

- 理解为什么需要端到端正确性测试
- 理解 golden logits 的作用
- 理解 cosine similarity 的比较方式

## 为什么这一步重要

单个模块正确不代表整条链路正确。端到端测试是确认“从输入到输出”都没问题的最直接方式。

## 这一页的核心内容

- 输入 prompt
- 跑完整推理链路
- 对照 PyTorch golden logits
- 使用 cosine similarity 评价结果

端到端正确性测试的作用是把前面章节各自正确的模块串起来，验证整体行为是否仍然正确。这里建议固定模型版本和输入，保证 golden logits 有可比性。

如果测试输入变了，golden 结果也要同步更新，否则对照就失去意义。

这一页可以把 golden logits 的角色写得更明确：它不是要求数值逐位完全一致，而是要求与参考实现保持足够接近。这样读者就不会把端到端测试误解成一个“必须绝对相等”的测试。

`cosine_similarity` 适合捕捉输出向量的整体方向一致性，但当 logits 存在系统性缩放偏置时，还需要补充 `mean_absolute_error` 或 `mse` 作为辅助指标，以确保数值量级也在可接受范围内。两者结合使用通常比单一指标更可靠。

端到端对照伪代码可以写成：

```cpp
auto golden = pytorch_forward(prompt);
auto actual = engine_forward(prompt);
EXPECT_GT(cosine_similarity(golden, actual), 0.99f);
```

## 验证标准

- cosine similarity 达到预期阈值
- 输出形状正确
- 结果稳定可复现

如果相同输入在多次运行中都落在同一误差范围内，说明这条链路已经足够稳定，可以作为回归基线。

## 常见问题

- golden logits 不匹配当前模型版本
- 输入预处理不一致
- 指标阈值和场景不匹配
