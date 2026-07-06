# 05.1 Embedding and Positional Encoding Embedding 与位置编码


对应代码：见 [本章代码索引](/code/chapter05_generation)

## 这一页要解决什么

把 token 变成 decoder 可以直接消费的向量输入，并把顺序信息注入进去。

## 2. 背景是什么

生成链路的第一步不是 attention，而是输入表示。没有这一步，后面的 Transformer 层没有基础输入，模型只能看到离散 token id，看不到词序。

## 3. 最小设计是什么

先把这一步拆成两个独立动作：

- token id -> embedding 向量
- position -> 位置编码

这样做的好处是，查表逻辑和顺序注入可以分别验证，也更容易和后续 decoder 对接。

### Embedding 查表怎么做

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

### 输入和输出的维度

- 输入 `token_ids`: `[B, S]`
- 输出 `embedding`: `[B, S, D]`

也就是说，token 还是那些 token，但每个 token 从一个整数 id 变成了一个长度为 `D` 的向量。

### 位置编码方案对比

| 方案 | 实现方式 | 优点 | 常见模型 |
| --- | --- | --- | --- |
| Sinusoidal | 固定正弦 / 余弦函数 | 无需学习，可外推 | 原始 Transformer |
| Learned | 可学习参数 | 灵活 | BERT |
| RoPE | 旋转位置编码 | 相对位置信息更自然 | LLaMA、GPT-NeoX |

本教程先用 **Sinusoidal** 作为基线，因为它实现简单、边界清楚。  
后续如果要切到 RoPE，主要替换的是位置编码实现，不需要改输入输出边界。

## 4. 中间推导是什么

如果从工程角度倒推，embedding 和位置编码最好分开写：

- 先确认 token id 到向量的映射正确
- 再确认顺序信息是否真的进入表示
- 如果后面切换到 RoPE，只需要替换位置编码的实现，不需要改输入输出边界

### 特殊 token 和 padding

生成场景里，embedding table 往往还包含特殊 token：

- `<s>`：序列开始
- `</s>`：序列结束
- `<pad>`：填充
- `<unk>`：未知 token

这些 token 也走同样的查表流程。  
`<s>` 常作为初始输入，`</s>` 常作为停止条件，`<pad>` 主要用于对齐 batch 中不同长度的序列。

最小伪代码可以写成：

```cpp
Tensor x = embedding_lookup(token_ids, embedding_table);
Tensor pos = position_encoding(token_ids.size(), hidden_size);
Tensor input = x + pos;
```

## 5. 怎么验证

- 输入 token 可以映射成向量
- 输出 shape 正确
- 同一个 token 在不同位置得到的表示不同
- `token_ids: [B, S]` 能变成 `embedding: [B, S, D]`

如果 shape 稳定、位置变化能影响表示，这一步就已经跑通了。

## 6. 常见问题

- embedding 维度和模型 hidden size 不一致
- 位置编码加错维度
- learned position 和 RoPE 混用

## 7. 和后续章节怎么接

这一页的输出会直接进入 `05.2 N-layer Transformer Assembly`。  
可以把整条链路理解成：

```text
05.1 Input Representation
  -> 05.2 N-layer Transformer Assembly
  -> 05.3 Naive Generation
```

只要输入表示稳定，后面的多层堆叠就可以继续沿着同一条 hidden state 传递。
