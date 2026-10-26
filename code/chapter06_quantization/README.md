# Chapter 06：Quantization 量化与 CPU/GPU 后端

本目录从量化 Tensor 和 Q4_K block 开始，建立 CPU reference、fused/AVX2 backend 和最小 benchmark；GPU 非融合基线位于相邻的 `code/cuda`。完整学习说明见 [`source/chapter06_quantization`](../../source/chapter06_quantization/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `src/dequantize_op.cpp` | 通用反量化算子 |
| `src/q4k_dequantize.cpp`、`src/q6k_dequantize.cpp`、`src/q8_0_dequantize.cpp` | Q4_K/Q6_K/Q8_0 block 解码 |
| `src/linear_layer.cpp`、`src/quantized_linear.cpp` | 量化 Linear 路径 |
| `src/q4k_kernel.cpp` | Q4_K fused CPU kernel |
| `src/q4k_linear.cpp` | Q4_K scalar reference 和 backend 适配 |
| `src/q4k_linear_avx2.cpp` | Q4_K fused AVX2/FMA 解码与点积 backend |
| `src/q6k_linear_avx2.cpp` | Q6_K fused AVX2/FMA 解码与点积 backend |
| `src/q8_0_linear.cpp` | Q8_0 reference、scalar fused 与 backend 适配 |
| `include/chapter06_quantization/quantized_linear_policy.h`、`src/quantized_linear_policy.cpp` | Q4_K/Q6_K/Q8_0 统一 backend policy、registry 与 dispatcher |
| `tests/chapter06_quantization_test.cpp` | dtype、多 block、误差、policy 和 backend 测试 |
| `examples/q4k_benchmark.cpp` | Q4_K baseline/fused/SIMD benchmark |
| `scripts/run.sh` | 隔离构建、CTest 和 demo 入口 |
| Chapter 01/06 examples | 真实 GGUF Q4_K/Q6_K 跨章验证 |
| `../cuda` Quantized Linear 资产 | `CudaQuantizedMatrix -> CUDA dequantize -> Float32 Workspace -> cuBLAS SGEMM` |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter06_quantization/scripts/run.sh
```

Q4_K/Q6_K 已完成 CPU block reference、scalar fused、AVX2/FMA fused backend，以及 GPU dequantize + SGEMM correctness 基线。Q8_0 已完成 CPU block reference、scalar fused、统一 policy 和真实 SmolVLM 32 层 Decoder/LM Head 对照；AVX2 与 GPU backend 尚未实现。`QuantizedLinearPolicy` 管理 CPU backend，GPU 路径由 `CudaExecutionContext` 编排。
