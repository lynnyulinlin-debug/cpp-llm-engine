# 05.1 Embedding and Positional Encoding Embedding 与位置编码


## 这一页要解决什么

把 token 变成 decoder 可以直接消费的向量输入，并把顺序信息注入进去。

## 阅读入口

- 前置阅读：[`chapter05 导读`](./intro.md)、[`chapter04_attention`](../chapter04_attention/intro.md)
- 源码直达：<a href="../_code/chapter05_generation/src/embedding.cpp"><code>embedding.cpp</code></a>、<a href="../_code/chapter05_generation/src/position_encoding.cpp"><code>position_encoding.cpp</code></a>、<a href="../_code/chapter05_generation/tests/chapter05_generation_test.cpp"><code>chapter05_generation_test.cpp</code></a>
- 相关阅读：[`05.2 N 层 Transformer 组装`](./02_transformer_assembly.md)、[`05.3 Naive Generation`](./03_naive_generation.md)

## 核心设计

生成链路的第一步不是 attention，而是输入表示。没有这一步，后面的 Transformer 层没有基础输入，模型只能看到离散 token id，看不到词序。

先把这一步拆成两个独立动作：token id -> embedding 向量，position -> 位置编码。这样查表逻辑和顺序注入可以分别验证，也更容易和后续 decoder 对接。

Embedding 查表：

Embedding table 本质上是一个二维表：

- 形状通常是 `[vocab_size, hidden_size]`
- 每一行对应一个 token 的向量表示

查表操作本质上就是按 `token_id` 取出对应那一行，再把它放进输出 tensor 里。  
在 C++ 里，这通常是一次按行拷贝，而不是复杂的数值计算。

```cpp
class Embedding {
    std::vector<float> table_;  // vocab_size * hidden_size
public:
    // 伪代码：根据 token_id 取出一行向量
    Tensor lookup(int token_id) const;
};
```

输入和输出的维度：

- 输入 `token_ids`: `[B, S]`
- 输出 `embedding`: `[B, S, D]`

也就是说，token 还是那些 token，但每个 token 从一个整数 id 变成了一个长度为 `D` 的向量。

位置编码方案对比：

| 方案 | 实现方式 | 优点 | 常见模型 |
| --- | --- | --- | --- |
| Sinusoidal | 固定正弦 / 余弦函数 | 无需学习，可外推 | 原始 Transformer |
| Learned | 可学习参数 | 灵活 | BERT |
| RoPE | 旋转位置编码 | 相对位置信息更自然 | LLaMA、GPT-NeoX |

本教程先用 **Sinusoidal** 作为基线，因为它实现简单、边界清楚。  
后续如果要切到 RoPE，主要替换的是位置编码实现，不需要改输入输出边界。

这一层在 runtime 里怎么走：

这一节不是只在讲“embedding 是什么”，而是在定义 runtime 的输入管道：请求先把 prompt 编成 token ids，token ids 进入 embedding table 变成基础向量，位置编码把顺序信息加进去，最后送入 `Transformer` 的第一层。

最小的数据流可以写成：

```cpp
Tensor token_ids = request.prompt_ids();
Tensor x = embedding_lookup(token_ids, embedding_table);
Tensor pos = position_encoding(token_ids.size(), hidden_size);
Tensor hidden = x + pos;
hidden = transformer.forward(hidden);
```

这里的关键不是公式本身，而是边界：`request.cpp` 提供 prompt ids 和请求状态，`runtime.cpp` 把输入送到模型，`transformer.cpp` / `decoder_layer.cpp` 消费已经带有位置信息的 hidden state。这一页真正解决的是“输入怎样从请求状态变成模型状态”，而不是单纯解释 embedding 查表。

如果从工程角度倒推，embedding 和位置编码最好分开写：先确认 token id 到向量的映射正确，再确认顺序信息是否真的进入表示；如果后面切换到 RoPE，只需要替换位置编码实现，不需要改输入输出边界。

特殊 token 和 padding：

生成场景里，embedding table 往往还包含特殊 token：`<s>`、`</s>`、`<pad>`、`<unk>`。这些 token 也走同样的查表流程，其中 `<s>` 常作为初始输入，`</s>` 常作为停止条件，`<pad>` 主要用于对齐 batch 中不同长度的序列。

最小伪代码可以写成：

```cpp
Tensor x = embedding_lookup(token_ids, embedding_table);
Tensor pos = position_encoding(token_ids.size(), hidden_size);
Tensor input = x + pos;
```

## 验证与衔接

验证时重点看：

- 输入 token 可以映射成向量
- 输出 shape 正确
- 同一个 token 在不同位置得到的表示不同
- `token_ids: [B, S]` 能变成 `embedding: [B, S, D]`

如果 shape 稳定、位置变化能影响表示，这一步就已经跑通了。

常见问题：

- embedding 维度和模型 hidden size 不一致
- 位置编码加错维度
- learned position 和 RoPE 混用

和后续章节怎么接：

这一页的输出会直接进入 `05.2 N-layer Transformer Assembly`。可以把整条链路理解成：

```text
05.1 Input Representation
  -> 05.2 N-layer Transformer Assembly
  -> 05.3 Naive Generation
```

只要输入表示稳定，后面的多层堆叠就可以继续沿着同一条 hidden state 传递。
