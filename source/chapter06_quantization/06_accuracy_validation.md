# 06.6 Accuracy Validation 精度验证


## 这一页要解决什么

这一页要回答一个很关键的问题：量化模型和 FP16 模型之间的差异，是否还在可接受范围内？

量化的价值不是“更小”本身，而是“在可接受精度下更高效”。  
没有精度验证，就无法判断当前方案是否真的适合部署。

## 阅读入口

- 前置阅读：[`06.5 Quantized Linear`](./05_linear_with_quant.md)、[`06.4 DequantizeOp`](./04_dequantize_op.md)
- 源码直达：<a href="../_code/chapter06_quantization/tests/chapter06_quantization_test.cpp"><code>chapter06_quantization_test.cpp</code></a>、<a href="../_code/chapter06_quantization/src/linear_layer.cpp"><code>linear_layer.cpp</code></a>、<a href="../_code/chapter06_quantization/src/quantized_linear.cpp"><code>quantized_linear.cpp</code></a>
- 相关阅读：[`chapter06 小结`](./summary.md)、[`chapter07_benchmark`](../chapter07_benchmark/intro.md)

## 验证设计

精度验证的核心，不是简单跑一个 pass/fail，而是把“量化节省了什么、牺牲了什么”清楚表达出来。  
对真实部署来说，能接受的误差阈值往往和任务、模型、硬件有关，所以这一页的重点是建立方法，而不是写死一个数字。

建议把验证结果拆成两层：

- 一层看局部 logits 或中间 tensor
- 一层看最终输出

这样既能判断量化策略是否整体可用，也能在失败时快速定位是哪一层开始偏离。

局部验证的锚点通常选择在量化路径与 FP16 路径共享接口的位置，比如 LayerNorm 输出、Attention 输出或 FFN 输出。在这些位置插入比较点，可以判断误差是在特定模块内部积累，还是在跨模块传播中被放大。

对照方式：

最小对照伪代码可以写成：

```python
golden = fp16_model(x)
quant = quant_model(x)
score = cosine_similarity(golden, quant)
```

这足以表达“比较什么、怎么比较”。

## 验证与排查

验证时重点看：

- 使用相同输入
- 比较输出向量或 logits
- 观察是否达到阈值

如果能在多个输入样本上都保持相近的误差区间，就说明这条量化路径的稳定性更可信。

常见问题：

- 只看单个样本
- 指标阈值设置不合理
- 没有区分局部误差和整体误差
