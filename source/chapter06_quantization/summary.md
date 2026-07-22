# 06.8 Chapter Wrap-up 本章解决了什么

这章回答的是：量化链路是否已经能工作，并且是否还能保住可接受精度？到这里，INT8 权重、反量化路径和精度验证已经连成一条可解释的链路。

## 1. 本章解决了什么

- INT8 量化链路已经能工作
- 反量化路径已经清晰可验证
- 量化与精度验证已经同时落地

## 生态回看

回看 `TensorRT-LLM`、`Gemma` 和 `vLLM` 时，有助于看懂“量化链路”在真实生态里的三个落点。

- 先看 [06.7 TensorRT-LLM / Gemma / vLLM 对照](/chapter06_quantization/07_tensorrt_gemma_vllm_comparison)
- 对照 [TensorRT-LLM](/reference/ecosystem/tensorrt_llm) 回看 engine 和部署路径
- 对照 [Gemma](/reference/ecosystem/gemma) 回看模型家族和误差判断
- 对照 [vLLM](/reference/ecosystem/vllm) 回看量化后的 serving runtime 行为

这三点连起来，说明量化不再只是“能不能加载”，而是“加载后的链路还能不能被可靠解释”。

## 阅读入口

- 前置阅读：[`chapter06 导读`](/chapter06_quantization/intro)、[`06.5 Quantized Linear`](/chapter06_quantization/05_linear_with_quant)
- 源码直达：[`chapter06_quantization 代码入口`](/code/chapter06_quantization)
- 相关阅读：[`06.7 TensorRT-LLM / Gemma / vLLM 对照`](/chapter06_quantization/07_tensorrt_gemma_vllm_comparison)、[`chapter07_benchmark`](/chapter07_benchmark/intro)

## 代码直达

- [`src/tensor.cpp`](/code/chapter06_quantization#tensor-cpp)
- [`src/dequantize_op.cpp`](/code/chapter06_quantization#dequantize-op-cpp)
- [`src/linear_layer.cpp`](/code/chapter06_quantization#linear-layer-cpp)
- [`src/quantized_linear.cpp`](/code/chapter06_quantization#quantized-linear-cpp)
- [`tests/chapter06_quantization_test.cpp`](/code/chapter06_quantization#chapter06-quantization-test-cpp)

## 2. 为什么它重要

这里要回答的不只是“量化链路能不能工作”，还包括“省下来的资源成本是否值得这部分数值近似”。

## 3. 背景回顾

`chapter05` 已经提供了稳定的 generation runtime，这一章就在这条 runtime 上判断量化是否值得继续做。

## 4. 中间推导回顾

从“怎样用更低成本继续跑同一条链路”倒推，最自然的结果就是先从权重开始，再把反量化和算子路径一起纳入 runtime。中间任何一步断了，后面的结论都不成立。

## 5. 学习后应能做什么

- 看懂量化权重如何进入推理引擎
- 说明反量化在 runtime 中的位置
- 用精度对照判断量化实现是否可接受

## 6. 边界说明

这一章不负责把所有低比特方案都做完，只负责把量化变成一条可解释、可验证、可继续优化的部署路径。

在实践里，"值得继续"通常会同时看三条基线：精度损失是否低于任务容忍阈值、内存或带宽节省是否达到目标、推理速度提升是否满足部署需求。

## 代码回看

- [`src/linear_layer.cpp`](/code/chapter06_quantization#linear-layer-cpp)
- [`src/dequantize_op.cpp`](/code/chapter06_quantization#dequantize-op-cpp)
- [`src/quantized_linear.cpp`](/code/chapter06_quantization#quantized-linear-cpp)
- [`tests/chapter06_quantization_test.cpp`](/code/chapter06_quantization#chapter06-quantization-test-cpp)

## 7. 下一步是什么

进入 `chapter07_benchmark`，把正确性验证和性能测量体系补完整。

## 8. Advanced Notes

想继续深入的读者，可以继续思考下面这些问题：

- per-channel 和 groupwise quantization 会怎样影响精度？
- INT8、INT4 和 FP8 这几条路径在工程上有什么实际差异？
- 什么时候量化和反量化可以融合进相邻算子？
- calibration 数据应该怎样影响最终部署决策？

## 9. 收口结论

这一章留下的是一个“是否值得继续”的判断起点：先确认量化带来的成本下降和误差边界，再决定后续 benchmark 应该重点测什么。只有这个条件成立，`chapter07` 的测试和 benchmark 才不是空测，而是对一条已经可用的链路做回归和性能验证。
