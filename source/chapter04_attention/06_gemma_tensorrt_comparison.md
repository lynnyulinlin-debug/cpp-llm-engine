# 04.6 Decoder 成熟框架对照

本章把权重、Tensor view 和 MatMul 组合成单层 `DecoderLayer`。对照成熟项目时，先核对模型层的数据流，再观察 Attention backend、KV metadata 和 GPU kernel 如何接入。

## 对象映射

Gemma 提供具体模型家族，nano-vLLM 提供紧凑算法骨架，vLLM 和 TensorRT-LLM 展示生产执行路径。

| 教程对象 | 成熟项目中的对应物 | 阅读重点 |
| --- | --- | --- |
| `DecoderLayer` | Gemma/Llama decoder block | Norm、Attention、Residual、FFN 的顺序和结构差异 |
| Q/K/V projection | model attention 输入准备 | head 数、GQA layout、position 和权重 shape |
| `AttentionBackend` | vLLM attention backend、TensorRT-LLM kernel | mask、cache metadata 与 kernel 的接口 |
| 连续 `KvCache` | paged/block KV cache 的数值基础 | prefill 和 decode 怎样复用 K/V |
| `DecoderLayerContract` | model config 与 runner 约束 | Runtime 如何检查模型结构能力 |

## 教学实现与生产实现

教学实现固定单层数据流和 CPU 数值基线；成熟系统在相同语义上增加模型注册、分页缓存、融合 kernel 和并行执行。

| 本章可验证对象 | 成熟系统继续增加 | 后续入口 |
| --- | --- | --- |
| 单层 Pre-Norm Decoder | 多模型架构注册和多层执行计划 | Chapter01、Engine |
| CPU causal attention 与稳定 softmax | FlashAttention、fused attention、GPU workspace | CUDA backend |
| position offset 与连续 KV | paged KV、prefix reuse 和淘汰 | Chapter05、Chapter09 |
| 单层结构和数值测试 | tensor/pipeline parallel、跨设备通信 | 分布式扩展 |

## 源码阅读问题

阅读项目卡中的源码入口时，沿 `model layer -> attention -> backend` 追踪一次 forward：

1. model layer 为 Q/K/V、position 和 residual 准备了哪些输入？
2. Attention backend 接收数值 Tensor，还是同时接收 cache 与调度 metadata？
3. cache shape 由模型结构决定，物理分配又由哪个运行时对象决定？
4. prefill 和 decode 复用同一层权重时，输入 shape 与 metadata 有何差异？

| 项目卡 | 在本章中的用途 |
| --- | --- |
| [Gemma](../reference/ecosystem/gemma.md) | 对照具体模型家族的 Decoder 结构 |
| [nano-vLLM](../reference/ecosystem/nano_vllm.md) | 先阅读紧凑的模型与 forward 控制流 |
| [vLLM](../reference/ecosystem/vllm.md) | 观察 model layer、Attention backend 和 cache metadata |
| [TensorRT-LLM](../reference/ecosystem/tensorrt_llm.md) | 观察 GPU Runtime、Attention 和融合 kernel |
