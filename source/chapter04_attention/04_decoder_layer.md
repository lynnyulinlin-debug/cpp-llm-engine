# 04.4 DecoderLayer 解码器层


## 这一页要解决什么

把 Pre-LN、Attention、Residual 和 FFN 组合成一个完整的 Transformer decoder layer。

## 阅读入口

- 前置阅读：[`04.1 Multi-head Attention`](/chapter04_attention/01_multihead_attention)、[`04.3 Feed Forward Network`](/chapter04_attention/03_feedforward)
- 源码直达：[`chapter04_attention 代码入口`](/code/chapter04_attention)
- 相关阅读：[`04.5 首层权重加载`](/chapter04_attention/05_load_first_layer)、[`04.7 本章小结`](/chapter04_attention/summary)

## 读完你应该能做什么

- 理解 DecoderLayer 的标准结构
- 理解残差连接在模块组合中的作用
- 理解模块组合后如何继续保持 shape 一致

## 为什么这一步重要

单个模块正确不代表整体正确。DecoderLayer 是各模块真正拼接起来的地方，也是后面多层 Transformer 的最小单元。

## 标准结构

- Pre-LN
- Attention
- Residual
- FFN
- Residual

这里的重点不是“把模块堆起来”，而是保持各模块之间的 shape、顺序和数据流都一致。只要其中一个环节顺序错了，模型就会看起来能跑，但结果明显不对。

建议把 DecoderLayer 当作一个独立的中间里程碑：它既够小，方便验证；又够完整，足以代表一层 Transformer 的核心行为。

### 为什么使用 Pre-LN

Transformer 常见有两种归一化位置：

- Post-LN：`y = LayerNorm(x + sublayer(x))`
- Pre-LN：`y = x + sublayer(LayerNorm(x))`

本教程使用的是 **Pre-LN**，原因是：

1. 训练更稳定，梯度更容易通过残差路径传播
2. 收敛通常更顺滑
3. 现代大模型里 Pre-LN 更常见

对推理实现来说，Pre-LN 也更容易把模块边界画清楚：先归一化，再进入子模块，再做残差加法。

### 单层参数量的粗略估算

假设：

- `hidden_dim = D`
- `ffn_dim = 4D`

那么一层 DecoderLayer 的主要参数量可以粗略看成：

- Attention：`4 * D^2`，对应 Q/K/V/O 四个投影矩阵
- FFN：`2 * D * 4D = 8D^2`
- LayerNorm：两个归一化，约 `4D`
- 合计：`≈ 12D^2 + 4D`

如果 `D = 768`，那单层大约就是几百万级参数。这个估算后面在 benchmark 和 quantization 章节会反复用到。

### 从单层到多层

完整 Transformer 不是一层 DecoderLayer，而是多层堆叠：

```python
y = x
for i in range(num_layers):
    y = decoder_layer_i(y)
```

每一层的参数独立，但结构相同。层数越多，表达能力越强，代价也越高。

## 实现建议

- 先把每个子模块单独验证过
- 再组合成完整层
- 每一步都检查 shape

## 怎么验证

- 输入输出 shape 一致
- 残差连接不破坏维度
- 模块顺序符合预期

如果你能对一层 DecoderLayer 画出完整的数据流图，并让代码和图一致，这一页就完成了。

最小伪代码可以写成：

```cpp
Tensor y = x;
y = y + attention(layernorm1(y));
y = y + ffn(layernorm2(y));
```

源码锚点：`code/chapter04_attention/src/decoder_layer.cpp`

这段代码把 Pre-LN、Residual 和两个子模块的组合关系表达得很直接。

## 常见问题

- 残差连接的位置错误
- 归一化顺序错误
- 子模块输出 shape 与输入不匹配
- 先 Residual 再 LayerNorm，容易让训练和推理的行为偏离预期
- 模块顺序颠倒时，常见现象是 shape 看起来对，但结果明显不对
