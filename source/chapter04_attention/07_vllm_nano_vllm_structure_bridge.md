# 04.7 Decoder 结构与成熟框架阅读桥

Chapter01-03 的权重、view 和 MatMul 在本章汇合成 DecoderLayer。对照成熟框架时，先追模型层的 forward，再看 runtime 传入的位置和 cache metadata。

## 职责映射

| 教程对象 | 成熟框架中的对应物 | 阅读重点 |
| --- | --- | --- |
| `DecoderLayer` | 模型实现中的 decoder block | Norm、Attention、Residual、FFN 的顺序 |
| Q/K/V projection | attention 输入准备 | head 数、GQA layout 和 position 怎样进入 |
| `AttentionBackend` | FlashAttention、paged attention 或 fused attention | mask、cache layout 与 kernel 如何解耦 |
| 连续 `KvCache` | paged/block KV cache 的数值基础 | K/V 在 prefill 和 decode 间怎样复用 |
| `DecoderLayerContract` | model config 与 runner 输入约束 | 模型结构能力如何被 runtime 检查 |

## 源码入口

先选一个 Llama layer，沿 `forward -> attention -> backend` 阅读；scheduler 属于 Chapter05，不在这一遍混读。

| 框架 | 官方入口 | 带着什么问题读 |
| --- | --- | --- |
| vLLM | [`llama.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py) | 一层 Llama 怎样装配 attention、MLP 和权重 |
| vLLM | [`attention`](https://github.com/vllm-project/vllm/tree/main/vllm/attention) | model layer 怎样把 metadata 交给 attention backend |
| TensorRT-LLM | [`kernels`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels) | attention 与 fused kernel 在哪里落地 |

## 教程保留与省略

| 已实现 | 暂不实现 | 后续位置 |
| --- | --- | --- |
| 单层 Pre-Norm Decoder 数据流 | 多模型架构注册 | Chapter01 模型映射扩展 |
| CPU causal attention 与稳定 softmax | FlashAttention 和 GPU fused attention | GPU backend |
| position offset 与连续 KV | paged KV、prefix reuse、淘汰 | Chapter05/09 |
| 单层结构和数值测试 | tensor/pipeline parallel | 分布式扩展 |

## 源码阅读问题

1. model layer 负责 Q/K/V 和 position 的哪些准备，backend 接收什么？
2. cache 的 shape 由模型结构决定，还是由 scheduler 决定？
3. prefill 和 decode 复用同一层权重时，输入 metadata 有什么不同？
