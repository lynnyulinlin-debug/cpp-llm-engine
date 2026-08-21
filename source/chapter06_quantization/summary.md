# 06.9 Chapter Wrap-up 本章解决了什么

这章回答的是：量化链路是否已经能工作，并且是否还能保住可接受精度？到这里，INT8 权重、反量化路径和精度验证已经连成一条可解释的链路。

## 本章解决了什么

- INT8 量化权重已经能进入 loader 和 Tensor 表示
- 反量化路径已经清晰可验证
- 量化 Linear 已经能接回原有推理链路
- 精度验证已经能对照 FP16 基线
- 量化已经能放回 serving runtime 的 loader、kernel、cache 和 benchmark 边界里理解

## 为什么重要

这一章不只是证明“int8 能加载”，而是把量化放回部署判断里：它省了哪部分内存或带宽，误差来自哪里，是否值得继续进入 benchmark。

## 下一步

进入 `chapter07_benchmark`，把正确性验证和性能测量体系补完整。`chapter06` 留下的是量化链路，`chapter07` 要判断这条链路是否真的值得保留。

## 复盘路径

- 先看 [`06.7 TensorRT-LLM / Gemma / vLLM 对照`](./07_tensorrt_gemma_vllm_comparison.md)
- 再看 [`06.8 量化在 Serving Runtime 里的位置`](./08_quantization_in_serving_runtime.md)
- 对照 [`TensorRT-LLM`](../reference/ecosystem/tensorrt_llm.md) 回看 engine 和部署路径
- 对照 [`Gemma`](../reference/ecosystem/gemma.md) 回看模型家族和误差判断
- 对照 [`vLLM`](../reference/ecosystem/vllm.md) 回看量化后的 serving runtime 行为

这一章不负责把所有低比特方案都做完，也不负责证明量化一定更快。它只负责把权重量化变成一条可解释、可验证、可继续优化的路径。

在实践里，“值得继续”通常要同时看三条基线：精度损失是否低于任务容忍阈值、内存或带宽节省是否达到目标、推理速度提升是否满足部署需求。

- 先看 <a href="../_code/chapter06_quantization/src/tensor.cpp"><code>tensor.cpp</code></a>
- 再看 <a href="../_code/chapter06_quantization/src/dequantize_op.cpp"><code>dequantize_op.cpp</code></a>
- 然后看 <a href="../_code/chapter06_quantization/src/quantized_linear.cpp"><code>quantized_linear.cpp</code></a>
- 最后看 <a href="../_code/chapter06_quantization/tests/chapter06_quantization_test.cpp"><code>chapter06_quantization_test.cpp</code></a>
