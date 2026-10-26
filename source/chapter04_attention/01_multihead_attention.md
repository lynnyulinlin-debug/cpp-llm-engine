# 04.1 Multi-head Attention 多头注意力
## 学习入口

实现多头注意力的前向计算，并理解 Q/K/V、score、softmax 和输出合并的完整数据流。

读完后应能做到：

- 理解注意力公式的基本结构
- 理解 Q/K/V 线性变换的作用
- 理解每个 head 如何独立计算 score
- 理解 softmax 和输出合并的过程

- 前置阅读：[`chapter04 导读`](./intro.md)、[`MatMulOp 设计`](../chapter03_matmul/02_matmul_design.md)
- 源码直达：<a href="../_code/chapter04_attention/src/attention.cpp"><code>attention.cpp</code></a>、<a href="../_code/chapter04_attention/src/linear_layer.cpp"><code>linear_layer.cpp</code></a>、<a href="../_code/chapter04_attention/tests/chapter04_attention_test.cpp"><code>chapter04_attention_test.cpp</code></a>
- 相关阅读：[`04.2 LayerNorm 与激活`](./02_layernorm_and_activation.md)、[`04.4 DecoderLayer`](./04_decoder_layer.md)

## 核心数据流

注意力是 Transformer 的核心。它决定 token 之间如何相互作用，也决定后续 DecoderLayer 的表达能力。

从整条教程主线看，本节把 `chapter03` 的 Linear / MatMul 契约第一次放进真实模型结构里。后面读 vLLM / nano-vLLM 时，attention backend、KV cache 和 prefill / decode 路径的理解，都建立在这里固定的 Q/K/V 与 shape 关系上。

核心步骤：

- 输入做 Q/K/V 线性变换
- 按 head 切分
- 计算 attention score
- 进行 softmax
- 得到每个 head 的输出
- 合并所有 head 的结果

实现时最重要的是把张量 shape 一步一步对应起来。每一步输入和输出的维度都要能明确说清，这样后面调试才能定位到具体变换。

最小实现通常先从单 batch、单序列长度开始，然后再扩展到更一般的输入。

通用数学采用 `[B,S,D]` shape。仓库教学实现把 batch 固定为 1，因此 `Tensor` 和 `AttentionRequest` 使用 `[tokens, model_dim]` 连续 row-major 布局；Q、K、V head 数相同。TinyLlama 的 GQA、RoPE 和请求级分层 K/V 位于真实 CPU Context，不在该 `MultiHeadAttention` 类中。

| 能力 | 教学 `MultiHeadAttention` | TinyLlama CPU Context |
| --- | --- | --- |
| 输入布局 | `[tokens, model_dim]` | token hidden vectors |
| head 结构 | MHA，Q/K/V head 数相同 | GQA，32 个 Q heads、4 个 KV heads |
| 位置 | backend 接收 offset，类内固定为 0 | 每层 Q/K 应用 RoPE |
| K/V 历史 | 类内不持有；backend 测试可直接传历史 K/V | 每请求 22 层 K/V |

### 通用 shape

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

## 算法镜像与 C++ 映射

下面的 Python 只表达 CPU reference kernel 的循环顺序，不是第二套可运行实现。Q/K/V projection 和 head 合并由 `MultiHeadAttention::forward()` 处理；镜像从连续 `[token, head, dim]` buffer 开始。

```python
def attention_reference(query, key, value, heads, head_dim, causal, offset):
    output = zeros_like(query)
    scale = 1.0 / sqrt(head_dim)

    for query_token in range(len(query)):
        query_position = offset + query_token
        for head in range(heads):
            scores = []
            for key_token in range(len(key)):
                if causal and key_token > query_position:
                    scores.append(float("-inf"))
                    continue
                scores.append(dot(query[query_token][head], key[key_token][head]) * scale)

            probabilities = stable_softmax(scores)
            for key_token, probability in enumerate(probabilities):
                output[query_token][head] += probability * value[key_token][head]

    return output
```

| Python 动作 | C++ 位置 | 验证重点 |
| --- | --- | --- |
| `query_position = offset + query_token` | `query_position_offset` | decode 查询可以看到历史 K/V |
| causal 条件 | `key_token > query_position` | 未来位置写成负无穷 |
| `dot(...) * scale` | `dim` 内积循环 | head 之间不串数据 |
| `stable_softmax` | max、exp、denominator 三段循环 | 大分数输入不溢出 |
| 概率加权 `value` | output 聚合循环 | 输出保持 `[query_tokens, heads * head_dim]` |

## 验证与排查

验证时重点看：

- 输出 shape 正确
- 每个 head 的维度一致
- softmax 后的权重分布合理
- causal mask 和 `query_position_offset` 能正确访问历史 K/V

如果小样本输入下，attention 权重会对某些位置明显聚焦，而输出 shape 又保持稳定，通常说明这一步已经能工作。

常见问题：

- head 切分维度错误
- Q/K/V 线性变换维度不一致
- softmax 轴选错
