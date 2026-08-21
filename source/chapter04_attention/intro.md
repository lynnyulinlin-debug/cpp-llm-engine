# 04.0 Decoder Structure and Attention 组装 decoder block

这一章先回答一个结构问题：单层 decoder 要怎样做成一个可验证、可扩展的最小模块？

本章实现 Transformer decoder 的核心组件：多头注意力、LayerNorm、激活函数、FFN，并把它们组装成一个完整的 DecoderLayer。

从学习主线看，这一章是 `chapter01` 到 `chapter03` 的第一次结构汇合点：

- `chapter01` 提供权重怎样进入 runtime
- `chapter02` 提供 Tensor view 和 shared storage 语义
- `chapter03` 提供 MatMul 和 Linear 的稳定计算边界

这一章第一次把这些能力收束成一个真正的模型结构单元。后面的 `chapter05_generation` 不再先讨论“attention 公式”，而是直接调度这里已经闭合的 decoder 结构。

## 本章和 vLLM / nano-vLLM 的联系

在 vLLM 或 nano-vLLM 里，真实代码不会只停在一个孤立的 `DecoderLayer`，而是把它放进 model runner、attention backend 和 KV cache 写入路径里。本章先把 decoder block 的结构、shape、Q/K/V 和 position 边界讲清楚，后面读这些项目时才知道 runtime 调度的到底是哪一个模型结构，而不是把结构问题和调度问题混在一起。

## 本章概览

| 先看什么 | 这一节讲什么 | 为什么重要 | 对应小节 |
| --- | --- | --- | --- |
| Attention 怎么做 | Q/K/V、score、softmax 和输出合并 | 先建立 token 交互路径 | [`04.1 Multi-head Attention`](./01_multihead_attention.md) |
| LayerNorm 和激活做什么 | 归一化和非线性 | 先补齐 decoder 的基础组件 | [`04.2 LayerNorm and Activation`](./02_layernorm_and_activation.md) |
| FFN 怎么接入 | 两层线性和激活 | 先补齐逐 token 变换路径 | [`04.3 FeedForward`](./03_feedforward.md) |
| DecoderLayer 怎么闭合 | Attention、FFN、Residual 组合 | 先形成可验证单层结构 | [`04.4 DecoderLayer`](./04_decoder_layer.md) |
| 首层权重怎么接入 | 真实权重和 shape 验证 | 先确认结构能接真实模型 | [`04.5 Load First Layer`](./05_load_first_layer.md) |
| 怎么看真实生态 | Gemma 与 TensorRT-LLM 对照 | 先把结构放回工程坐标 | [`04.6 Gemma / TensorRT-LLM 对照`](./06_gemma_tensorrt_comparison.md) |
| 怎么衔接 serving runtime | DecoderLayer、Q/K/V、position 和 cache 边界 | 先建立 vLLM / nano-vLLM 阅读坐标 | [`04.7 vLLM / nano-vLLM 结构阅读桥`](./07_vllm_nano_vllm_structure_bridge.md) |
| 本章怎么收口 | 单层 decoder 和下一章入口 | 进入 runtime 前先固定结构边界 | [`04.8 本章小结`](./summary.md) |

## 阅读入口

- 前置阅读：[`chapter03_matmul`](../chapter03_matmul/intro.md)、[`chapter03 小结`](../chapter03_matmul/summary.md)
- 源码直达：<a href="../_code/chapter04_attention/src/attention.cpp"><code>attention.cpp</code></a>、<a href="../_code/chapter04_attention/src/decoder_layer.cpp"><code>decoder_layer.cpp</code></a>、<a href="../_code/chapter04_attention/tests/chapter04_attention_test.cpp"><code>chapter04_attention_test.cpp</code></a>
- 相关阅读：[`04.1 Multi-head Attention`](./01_multihead_attention.md)、[`04.4 DecoderLayer`](./04_decoder_layer.md)、[`04.5 首层权重加载`](./05_load_first_layer.md)、[`04.6 Gemma / TensorRT-LLM 对照`](./06_gemma_tensorrt_comparison.md)、[`04.7 vLLM / nano-vLLM 结构阅读桥`](./07_vllm_nano_vllm_structure_bridge.md)

## 这章在解决什么

这一章要做的是把前面做好的权重对象、view 语义和算子边界组合成一个单层 decoder，让它可以被单独验证、单独排错，也能继续往上扩展。重点是先把结构站稳，再谈生成 runtime。

## 这章怎么做

Transformer 的核心不是某一个独立算子，而是几个模块按固定顺序组合后的结构。最小设计就是把单层 decoder 做成一个稳定模块：输入 token embeddings，先经过 LayerNorm，再进入 Attention、Residual、FFN，最后回到 Residual。

如果从“怎样把基础算子拼成一个可验证的 decoder”倒推，结构就会自然收敛成 `Pre-LN -> Attention -> Residual -> FFN -> Residual`。

## 收口与下一步

这一章把注意力、FFN、LayerNorm 和残差稳定地收束成单层 decoder。验证时先看 shape，再看前向流程，最后看首层权重是否能接入。

下一步进入 `chapter05_generation`，把这些 decoder layer 放进一个可重复运行的生成循环里。`chapter03` 提供算子边界，`chapter04` 提供结构边界，`chapter05` 开始提供运行时边界。
