# 03.7 从 MatMul 到真实推理 Backend

本章用 `MatMulOp -> Backend -> Kernel` 分开计算语义和执行方式。成熟框架仍保留这三类职责，但会加入设备选择、算子融合、workspace 和多精度 kernel。

![Operator、Backend 与 Kernel 分层](../assets/diagrams/chapter03-operator-backend-kernel.svg)

## 职责映射

| 教程对象 | 成熟框架中的对应物 | 阅读重点 |
| --- | --- | --- |
| `MatMulOp` | Linear、QKV projection、FFN 的算子语义 | 输入 layout、shape 和输出是否稳定 |
| `MatmulBackend` | ggml backend、TensorRT-LLM runtime/kernel dispatch | 谁选择 CPU、GPU 或高性能库 |
| CPU reference kernel | scalar/reference implementation | 优化路径用什么结果做数值基线 |
| `code/cuda` cuBLAS backend | GEMM library、tactic 或 plugin | 库调用前后怎样处理 layout、stream 和 workspace |
| PyTorch 对照 | backend correctness test | backend 替换后允许多大误差 |

## 源码入口

先沿一个 Linear 或 MatMul 调用追到实际 kernel，再回看 backend 注册，不要从 kernel 目录逐文件浏览。

| 框架 | 官方入口 | 带着什么问题读 |
| --- | --- | --- |
| ggml / `llama.cpp` | [`ggml-backend.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp)、[`ggml-cpu`](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cpu) | graph op 怎样分派到 CPU backend |
| ggml / `llama.cpp` | [`ggml-cuda`](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda) | 同一 op 怎样选择 CUDA kernel 或库调用 |
| TensorRT-LLM | [`kernels`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels)、[`runtime`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/runtime) | kernel 能力怎样进入 engine runtime |

## 教程保留与省略

| 已实现 | 暂不实现 | 后续位置 |
| --- | --- | --- |
| row-major MatMul contract | transpose、batched GEMM、动态 layout | 扩展 Operator contract |
| CPU scalar reference、独立 cuBLAS backend | SIMD、batched GEMM 和自定义 fused kernel | CPU/GPU backend |
| backend 注入与可观察名称 | tactic 搜索、workspace planner | 性能优化阶段 |
| shape、buffer、数值回归与显式 stream | 多流 overlap、跨设备执行 | GPU 执行主线 |

## 源码阅读问题

1. 模型代码在哪里停止描述语义，backend 从哪里开始选择实现？
2. layout 转换发生在算子、backend 还是 kernel 内部，成本是否可见？
3. fused kernel 上线后，哪个 reference 路径继续承担正确性基线？
