# 04.1 Multi-head Attention 多头注意力


## 这一页要解决什么

实现多头注意力的前向计算，并理解 Q/K/V、score、softmax 和输出合并的完整数据流。

读完后应能做到：

- 理解注意力公式的基本结构
- 理解 Q/K/V 线性变换的作用
- 理解每个 head 如何独立计算 score
- 理解 softmax 和输出合并的过程

## 阅读入口

- 前置阅读：[`chapter04 导读`](./intro.md)、[`MatMulOp 设计`](../chapter03_matmul/02_matmul_design.md)
- 源码直达：<a href="../_code/chapter04_attention/src/attention.cpp"><code>attention.cpp</code></a>、<a href="../_code/chapter04_attention/src/linear_layer.cpp"><code>linear_layer.cpp</code></a>、<a href="../_code/chapter04_attention/tests/chapter04_attention_test.cpp"><code>chapter04_attention_test.cpp</code></a>
- 相关阅读：[`04.2 LayerNorm 与激活`](./02_layernorm_and_activation.md)、[`04.4 DecoderLayer`](./04_decoder_layer.md)

## 核心数据流

注意力是 Transformer 的核心。它决定 token 之间如何相互作用，也决定后续 DecoderLayer 的表达能力。

从整条教程主线看，这一页不是单独讲一个公式，而是在把 `chapter03` 的 Linear / MatMul 边界第一次放进真实模型结构里。后面读 vLLM / nano-vLLM 时，attention backend、KV cache 和 prefill / decode 路径之所以复杂，前提也是这里的 Q/K/V 与 shape 关系先被固定住。

核心步骤：

- 输入做 Q/K/V 线性变换
- 按 head 切分
- 计算 attention score
- 进行 softmax
- 得到每个 head 的输出
- 合并所有 head 的结果

实现时最重要的不是把公式背出来，而是把张量 shape 一步一步对应起来。每一步输入和输出的维度需要能明确说清，这样后面调试才不会变成“看值猜错在哪”。

最小实现通常先从单 batch、单序列长度开始，然后再扩展到更一般的输入。

head 切分前后的维度：

假设：

- `batch_size = B`
- `seq_len = S`
- `hidden_dim = D`
- `num_heads = H`
- `head_dim = D / H`

前置条件：`hidden_dim % num_heads == 0`，否则 head 无法平均切分。

线性变换后：

- `Q`, `K`, `V`: `[B, S, D]`

按 head 切分后，通常会先 `reshape` 再 `transpose`：

- `Q`, `K`, `V`: `[B, H, S, head_dim]`

这样每个 head 都能独立计算自己的 score：

- `score_head`: `[B, H, S, S]`
- `out_head`: `[B, H, S, head_dim]`

最后再把所有 head 合并回原始隐藏维度：

- `out`: `[B, S, D]`

mask 的作用位置：

在生成场景中，为了防止当前 token 看到未来 token，需要在 softmax 之前加入 mask：

```text
score = (Q @ K^T) / sqrt(head_dim)
score = score + mask
weight = softmax(score)
out = weight @ V
```

这里的 mask 通常是 `[S, S]` 的上三角矩阵，未来位置为 `-∞`，在多头场景下会广播到 `[B, H, S, S]`。

head 合并的逆操作：

`out_head: [B, H, S, head_dim]` 合并回去时，通常是：

1. `transpose` 到 `[B, S, H, head_dim]`
2. `reshape` 成 `[B, S, H * head_dim] = [B, S, D]`

实现建议：

- 先把数学流程讲清楚，再写代码
- 先实现单头，再扩展到多头
- 先保证 shape 正确，再关注性能

## 验证与排查

验证时重点看：

- 输出 shape 正确
- 每个 head 的维度一致
- softmax 后的权重分布合理

如果小样本输入下，attention 权重会对某些位置明显聚焦，而输出 shape 又保持稳定，通常说明这一步已经能工作。

一个简化的伪代码流程可以是：

```cpp
Tensor q = linear_q(x);
Tensor k = linear_k(x);
Tensor v = linear_v(x);
Tensor score = softmax((q @ transpose(k)) / sqrt(head_dim));
Tensor out = score @ v;
```

真正实现时还要加 head 拆分和合并，但这个最小流程足以表达核心数据流。

常见问题：

- head 切分维度错误
- Q/K/V 线性变换维度不一致
- softmax 轴选错
