# 04.3 Feed Forward Network 前馈网络
## 学习入口

实现 FFN 模块，用两层线性层和一个激活函数构成 Transformer block 的前馈部分。

读完后应能做到：

- 理解 FFN 为什么是两层线性层
- 理解中间维度扩展的作用
- 理解激活函数在 FFN 中的位置

- 前置阅读：[`chapter04 导读`](./intro.md)、[`04.2 LayerNorm 与激活`](./02_layernorm_and_activation.md)
- 源码直达：<a href="../_code/chapter04_attention/src/feed_forward.cpp"><code>feed_forward.cpp</code></a>、<a href="../_code/chapter04_attention/src/linear_layer.cpp"><code>linear_layer.cpp</code></a>、<a href="../_code/chapter04_attention/tests/chapter04_attention_test.cpp"><code>chapter04_attention_test.cpp</code></a>
- 相关阅读：[`04.4 DecoderLayer`](./04_decoder_layer.md)、[`04.5 首层权重加载`](./05_load_first_layer.md)

## 核心结构

FFN 是 Transformer 中与注意力并列的核心计算模块。注意力负责 token 之间的信息交互，FFN 负责单个 token 的非线性变换。

- 第一层线性映射
- 激活函数
- 第二层线性映射

教学实现先串起两层线性和一个激活函数，形成可测试的 DecoderLayer 输入；真实 TinyLlama 路径再用 gate/up/down 三条投影和 SwiGLU 替换它。

如果后面要做门控 FFN、SwiGLU 或其他变体，也可以在这个结构上扩展，而不是重新设计一套新接口。

中间隐藏维度的典型值：

在很多 Transformer 里，FFN 的中间隐藏维度通常是 `hidden_dim` 的 4 倍：

- `hidden_dim = 768`
- `ffn_dim = 3072 = 4 * 768`

这是一种常见经验值，能给中间表示提供足够空间，同时不把参数量推得太高。

不同模型会略有差异：

- BERT 通常是 4 倍扩展
- GPT-2 通常是 4 倍扩展
- T5 通常是 4 倍扩展
- LLaMA 常用 SwiGLU 变体，等效扩展倍数会不同

FFN 变体对比：

| 变体 | 结构 | 常见模型 |
| --- | --- | --- |
| 标准 FFN | `Linear -> Activation -> Linear` | BERT、GPT-2 |
| Gated FFN | `Linear -> Activation * Gate` | PaLM |
| SwiGLU | `Linear_value` 和 `Linear_gate` 配合，再做门控输出 | LLaMA、Mistral |

标准 FFN 的形式最适合先实现：

```text
out = Linear2(Activation(Linear1(x)))
```

SwiGLU 的思路可以理解为：

```text
gate = Linear_gate(x)
value = Linear_value(x)
out = Linear_out(SiLU(gate) * value)
```

Chapter04 的 `FeedForward` 实现标准两层 FFN：`fc1 -> ReLU -> fc2`。真实 TinyLlama 走三组权重：`SiLU(gate(x)) * up(x) -> down`；该路径位于 `RealLlamaCpuContext`，并复用 Chapter06 的 Q4_K/Q6_K Linear backend。

| 对照项 | Chapter04 教学 FFN | TinyLlama 真实 FFN |
| --- | --- | --- |
| 投影 | `fc1`、`fc2` | `gate`、`up`、`down` |
| 激活 | ReLU | SiLU |
| 门控乘法 | 无 | `SiLU(gate) * up` |
| 权重 | 教学浮点参数 | GGUF Q4_K/Q6_K |

### FFN 在 DecoderLayer 中的位置

DecoderLayer 的标准顺序是：

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

也就是说，FFN 紧跟在 Attention 之后，接收 Attention 输出后的表示，再做逐 token 的非线性变换。

Attention 和 FFN 的职责对比：

- Attention：混合 token 之间的信息，回答“谁和谁相关”
- FFN：变换每个 token 的表示，回答“这个 token 应该变成什么”

两者合在一起，才构成 Transformer 的完整计算模式。

实现建议：

- 保持和 `MatMul` 接口一致
- 先实现最基础的两层结构
- 再考虑后续优化

## 验证与排查

验证时重点看：

- 输入输出 shape 正确
- ReLU 位于两次 Linear 之间
- 能和 DecoderLayer 组合

如果 FFN 的输入输出维度和 attention 的残差路径都能对上，说明这层已经可复用了。

FFN 的最小伪代码可以写成：

```cpp
Tensor hidden = matmul(x, w1);
hidden = activation(hidden);
Tensor out = matmul(hidden, w2);
```

这就是最基础的“两层线性 + 非线性”结构。

常见问题：

- 中间隐藏维度设置错误
- 激活函数位置放错
- 线性层输入输出顺序搞混
