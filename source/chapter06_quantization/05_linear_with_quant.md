# 06.5 Quantized Linear 量化 Linear


## 让 Linear 消费量化权重

把反量化逻辑与 Linear 层结合，支持量化权重推理。

读完后应能做到：

- 理解为什么要把 dequant 融入 Linear
- 理解量化权重推理路径如何组织
- 理解如何在不破坏接口的前提下提升效率

**阅读入口**

- 前置阅读：[`06.4 DequantizeOp`](./04_dequantize_op.md)、[`06.3 Load Quantized Weights`](./03_load_quantized_weights.md)
- 源码直达：<a href="../_code/chapter06_quantization/src/quantized_linear.cpp"><code>quantized_linear.cpp</code></a>、<a href="../_code/chapter06_quantization/include/chapter06_quantization/quantized_linear_policy.h"><code>quantized_linear_policy.h</code></a>、<a href="../_code/chapter06_quantization/src/quantized_linear_policy.cpp"><code>quantized_linear_policy.cpp</code></a>、<a href="../_code/chapter06_quantization/tests/chapter06_quantization_test.cpp"><code>chapter06_quantization_test.cpp</code></a>
- 相关阅读：[`06.6 Accuracy Validation`](./06_accuracy_validation.md)、[`chapter07_benchmark`](../chapter07_benchmark/intro.md)

## 核心实现

除了 INT8 `QuantizedLinear`，本章提供 `Q4KLinearReference` 和 `Q4KLinearFused`：前者先解码再计算 `W * input`，后者通过 `q4k_kernel.cpp` 在读取 block 时直接累加 dot product。GPU 首版由 `CudaQuantizedMatrix` 保存 packed payload，CUDA kernel 解码到复用 Workspace，再调用 cuBLAS SGEMM。

在 x86-64 环境中，`Q4KLinearAvx2` 和 `Q6KLinearAvx2` 都直接从 block 解码权重并用 AVX2/FMA 累加，不创建完整 float 权重数组。格式正确性仍由独立 dequant reference 验证，fused backend 负责性能实验。

本节的实现顺序固定为：

```text
Scalar reference
    -> Fused scalar
    -> SIMD dot product
    -> Fused SIMD decode + dot product
    -> CUDA dequantize + SGEMM baseline
    -> CUDA fused candidate
```

每一步都必须和上一层做数值回归，再看 benchmark。不能因为 SIMD 代码更复杂，就跳过可解释的 scalar 基线。

真实 TinyLlama 的 `token_embd.weight` 已完成首行 smoke test：取一个 contiguous Q4_K row、使用全 1 输入执行 CPU reference Linear，确认真实 payload 可以进入矩阵计算路径。该测试不代表完整模型已经组装完成，也不代表当前路径具有生产吞吐。

如果每次都先单独反量化再做线性计算，性能和内存开销都会更高。把流程融合进 Linear 更符合推理实际。

量化 Linear 需要覆盖：

- 量化权重进入 Linear
- dequant 与矩阵乘法结合
- 保持外部接口不变

量化 Linear 的重点是把量化逻辑放进最常用的推理算子里，而不是做成独立的“后处理”。这样更接近真实工程，因为最终性能往往取决于融合程度，而不是单个算子本身。

如果接口不变，上层模块就可以无感切换量化和非量化路径。

教程写法上，先把“先 dequant 再 matmul”的版本作为正确性基线，再对比已经实现的 scalar fused 路径。这样读者能看到融合如何减少中间 float buffer，同时保留一个容易排错的 reference 实现。

CUDA 非融合基线已经实现。下一步融合通常有两种方向：使用支持目标量化格式的库 kernel，或者手写 kernel，在加载权重的同时完成反量化与乘加。Q4_K/Q6_K 不是普通 INT8 矩阵，不能仅把接口替换成 INT8 GEMM 就认为语义等价。无论选择哪种方式，上层 Linear 接口都保持不变。

非融合基线可以写成：

```cpp
Tensor w = dequantize(qweight);
Tensor y = matmul(x, w);
```

Q4_K AVX2 路径每次读取 8 个 packed bytes，分别提取低四位和高四位，应用当前 64-value group 的 scale/min 后立即做 FMA。Q6_K 路径则从 `ql`、`qh` 和 `scales` 组合 8 个六位权重，再执行相同的向量乘加。

`QuantizedLinearPolicy` 同时保存 Q4_K、Q6_K 和 Q8_0 的 backend 偏好，`compute_quantized_linear()` 根据 `LinearRequest::weight_format` 完成派发。Q8_0 当前只有 portable scalar fused；请求其尚不存在的 AVX2 实现会返回明确的 `Status`，不会静默回退。

下面的片段对应 `quantized_linear_policy.h` 的真实接口。`request` 是 core `LinearRequest`；AVX2 仅在 `registry.q4k_avx2` 指向可用 backend 时可选。

```cpp
chapter06_quantization::QuantizedLinearPolicy policy{
    chapter06_quantization::QuantizedLinearBackendKind::Avx2Fused,   // Q4_K
    chapter06_quantization::QuantizedLinearBackendKind::ScalarFused, // Q6_K
    chapter06_quantization::QuantizedLinearBackendKind::ScalarFused  // Q8_0
};
auto status = chapter06_quantization::compute_quantized_linear(
    request, policy, registry);
```

真实 Decoder 按作用域保存 Attention、FFN 和 LM Head policy。Q4_K gate/up、Q6_K down、Q4_K QKV/out 三组实验都复用同一个 dispatcher，只改变对应作用域的 policy；LM Head 与其他作用域保持 scalar。三组实验均判定 `KEEP`，正常生成和服务仍默认使用可移植 scalar。

实现建议：

- 先确保正确性
- 再比较 reference、scalar fused 和 SIMD dot-product 三条路径
- 不破坏已有 FP16 路径

| Q4_K/Q6_K 路径 | 中间权重 buffer | 计算方式 | 用途 |
| --- | --- | --- | --- |
| Reference | 完整 float 数组 | 先反量化，再标量点积 | 格式与数值基线 |
| Scalar fused | 无 | 按 block 解码并标量累加 | 默认可移植路径 |
| AVX2/FMA fused | 无 | 8 个值一组解码并向量乘加 | 显式性能实验 |
| CUDA baseline | Float32 Workspace | CUDA block 解码后调用 cuBLAS SGEMM | GPU correctness 与 fused A/B 基线 |

## 验证与排查

验证时重点看：

- 量化 Linear 可以运行
- 结果与非量化版本接近
- 接口行为与原 Linear 一致

调用方继续使用同一个 `Linear` 接口，同时内部选择量化权重路径，说明接口与执行实现已经分离。

常见问题：

- 量化权重和激活的路径混乱
- 先做性能优化，后发现语义错误
- 融合后不好排错
