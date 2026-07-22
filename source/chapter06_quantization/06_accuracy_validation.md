# 06.6 Accuracy Validation 精度验证


## 这一页要解决什么

这一页要回答一个很关键的问题：量化模型和 FP16 模型之间的差异，是否还在可接受范围内？

量化的价值不是“更小”本身，而是“在可接受精度下更高效”。  
没有精度验证，就无法判断当前方案是否真的适合部署。

## 阅读入口

- 前置阅读：[`06.5 Quantized Linear`](/chapter06_quantization/05_linear_with_quant)、[`06.4 DequantizeOp`](/chapter06_quantization/04_dequantize_op)
- 源码直达：[`chapter06_quantization 代码入口`](/code/chapter06_quantization)
- 相关阅读：[`chapter06 小结`](/chapter06_quantization/summary)、[`chapter07_benchmark`](/chapter07_benchmark/intro)

## 代码直达

- [`tests/chapter06_quantization_test.cpp`](/code/chapter06_quantization#chapter06-quantization-test-cpp)
- [`src/linear_layer.cpp`](/code/chapter06_quantization#linear-layer-cpp)
- [`src/quantized_linear.cpp`](/code/chapter06_quantization#quantized-linear-cpp)
- [`src/tensor.cpp`](/code/chapter06_quantization#tensor-cpp)

## 为什么必须做对照

精度验证的核心，不是简单跑一个 pass/fail，而是把“量化节省了什么、牺牲了什么”清楚表达出来。  
对真实部署来说，能接受的误差阈值往往和任务、模型、硬件有关，所以这一页的重点是建立方法，而不是写死一个数字。

建议把验证结果拆成两层：

- 一层看局部 logits 或中间 tensor
- 一层看最终输出

这样既能判断量化策略是否整体可用，也能在失败时快速定位是哪一层开始偏离。

局部验证的锚点通常选择在量化路径与 FP16 路径共享接口的位置，比如 LayerNorm 输出、Attention 输出或 FFN 输出。在这些位置插入比较点，可以判断误差是在特定模块内部积累，还是在跨模块传播中被放大。

## 对照方式

最小对照伪代码可以写成：

```python
golden = fp16_model(x)
quant = quant_model(x)
score = cosine_similarity(golden, quant)
```

这足以表达“比较什么、怎么比较”。

## 怎么验证

- 使用相同输入
- 比较输出向量或 logits
- 观察是否达到阈值

如果能在多个输入样本上都保持相近的误差区间，就说明这条量化路径的稳定性更可信。

## 常见问题

- 只看单个样本
- 指标阈值设置不合理
- 没有区分局部误差和整体误差
