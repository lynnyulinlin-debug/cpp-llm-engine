# 03.7 从 MatMul 到真实推理 backend

本页不展开 vLLM 或 nano-vLLM 的完整算子系统，只说明本章的 `MatMulOp` 在真实推理项目里对应什么。读这些项目时，先把“模型里需要做什么计算”和“底层用哪个 backend 执行”分开。

## 阅读入口

- 前置阅读：[`03.2 MatMulOp Design`](./02_matmul_design.md)、[`03.4 cuBLAS Implementation`](./04_cublas_implement.md)
- 相关阅读：[`04.1 Multi-head Attention`](../chapter04_attention/01_multihead_attention.md)、[`04.4 DecoderLayer`](../chapter04_attention/04_decoder_layer.md)、[`vLLM 生态对照`](../reference/ecosystem/vllm.md)

## 本章内容对应到哪里

| 本章概念 | 在 vLLM / nano-vLLM 阅读中对应什么 | 先看懂什么 |
| --- | --- | --- |
| `Operator` | 上层调用算子的统一边界 | 模型代码不应该关心每个 backend 的细节 |
| `MatMulOp` | Linear、QKV projection、FFN 的核心计算 | 多数 decoder 计算最终会落到矩阵乘法 |
| CPU reference | 正确性基线 | 优化前要先知道语义是否算对 |
| cuBLAS backend | 高性能 GEMM 路径 | 同一语义可以换不同实现 |
| PyTorch 对照 | 数值回归基线 | 后端替换不能改变可接受的数值行为 |

## 为什么这页放在 chapter03

`chapter04` 会把 `MatMul` 用进 attention 和 FFN，但真实项目通常不会在模型结构里手写朴素矩阵乘法。它们会把 Linear、attention projection 或 fused op 分派给具体 backend，例如 cuBLAS、自定义 CUDA kernel、Triton kernel 或其他 runtime kernel。

所以本章要先固定一个边界：上层说清楚要算什么，下层负责决定怎么高效地算。

把前面 1 到 6 节压成一条最短链路，就是：

`Operator -> MatMulOp -> CPU reference -> cuBLAS backend -> PyTorch regression`

这条链路的意义是，先把算子语义立住，再把 backend 接上，最后用对照测试确认实现替换没有把语义改坏。读真实项目时，也可以沿着这条顺序去找对应层。

## 从教程到真实项目的三层对应

可以把本章内容压成一条更直接的工程映射：

- 教程里的 `MatMulOp`：真实项目里的稳定算子语义边界
- 教程里的 CPU / cuBLAS 切换：真实项目里的 backend dispatch
- 教程里的测试对照：真实项目里替换实现时的数值回归底线

这样读的时候，就不会把“Linear 在做什么”和“底层 kernel 怎样执行”混成同一个问题。

## 读源码时先抓三条线

第一，看 Linear 或 projection 的入口。Q/K/V projection、attention output projection 和 FFN 的升维 / 降维，本质上都离不开矩阵乘法语义。

第二，看 backend 分派。真实项目可能不会叫 `MatMulOp`，但通常会有某种 kernel 选择、backend dispatch 或 fused op 入口。

第三，看验证方式。只要 backend 可以替换，就必须保留数值对照和 shape contract，否则性能优化会变成不可解释的黑盒。

本教程当前用 CPU reference 和 cuBLAS 表达这条思想：先让算子语义稳定，再让实现可替换。后面进入 `chapter04_attention`，这个边界会被直接复用到 Q/K/V、attention output 和 FFN。

## 学完这一章后怎么看 vLLM / nano-vLLM

建议按这个顺序读，而不是一上来就钻进 kernel 细节：

1. 先找 Linear、QKV projection、FFN 这些计算入口
2. 再找 backend dispatch 或 fused path 的选择点
3. 最后再看 vendor library、自定义 kernel 或更细的实现优化

如果第二步还没看明白，就不要急着看第三步。否则很容易只看到性能技巧，却看不清这些技巧到底挂在哪个计算边界上。
