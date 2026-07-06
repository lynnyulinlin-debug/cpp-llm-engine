# 04.2 LayerNorm and Activation LayerNorm 与激活函数


## 这一页要解决什么

实现 Transformer 中常用的 LayerNorm 和激活函数，为后续 FFN 和 DecoderLayer 提供基础模块。

## 阅读入口

- 前置阅读：[`chapter04 导读`](/chapter04_attention/intro)、[`04.1 Multi-head Attention`](/chapter04_attention/01_multihead_attention)
- 源码直达：[`chapter04_attention 代码入口`](/code/chapter04_attention)
- 相关阅读：[`04.3 Feed Forward Network`](/chapter04_attention/03_feedforward)、[`04.4 DecoderLayer`](/chapter04_attention/04_decoder_layer)

## 读完你应该能做什么

- 理解 LayerNorm 的作用
- 理解激活函数在非线性表达中的意义
- 理解 SiLU 和 GELU 的使用场景

## 为什么这一步重要

LayerNorm 稳定训练和推理中的数值分布，激活函数引入非线性表达能力。它们虽然不是最“显眼”的模块，但对 Transformer 的稳定性很关键。

## 这一页的核心内容

- LayerNorm 的均值和方差归一化
- 可学习缩放和偏移参数
- SiLU / GELU 激活函数

LayerNorm 的实现重点是归一化维度和数值稳定性，激活函数的重点是公式一致性。两者都很容易写成“看起来对”，但细节不对的版本，所以最好先用小输入验证。

工程上建议把 LayerNorm 的参数、统计量计算和输出变换分开写清楚，这样后续复用更简单。

### LayerNorm 的归一化维度

Transformer 里的 LayerNorm 是对每个 token 的特征维度单独归一化：

- 输入: `[B, S, D]`
- 归一化维度: 最后一维 `D`
- 输出: `[B, S, D]`

也就是说，LayerNorm 不改变 shape，只改变每个 token 的特征分布：

```text
x_shape = [B, S, D]
mean = x.mean(dim=-1, keepdim=True)   # [B, S, 1]
var  = x.var(dim=-1, keepdim=True)    # [B, S, 1]
y = (x - mean) / sqrt(var + eps) * gamma + beta   # [B, S, D]
```

这也是 LayerNorm 和 BatchNorm 的关键区别：

- BatchNorm 通常在 batch 维度上统计
- LayerNorm 在特征维度上统计

### SiLU 和 GELU

SiLU (Swish)：

```text
SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
```

GELU (Gaussian Error Linear Unit)：

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

在 Transformer 里，两者都很常见：

- BERT 更常见 GELU
- LLaMA 更常见 SiLU
- GPT 系列也常见 GELU

教程实现时，最好把激活函数做成可配置的纯函数，后面 FFN 直接复用。GELU 如果想先做最小版本，也可以用常见近似式替代精确的 `erf` 计算。

### 推理中的一致性

LayerNorm 在训练和推理里的行为是一致的：

- 训练：按当前输入计算均值和方差
- 推理：仍然按当前输入计算均值和方差

所以它不需要像 BatchNorm 那样维护 running mean / running var。对本教程来说，这意味着同一份 LayerNorm 实现可以直接用于推理路径。

## 实现建议

- 先实现基础公式
- 再关注数值稳定性
- 保持接口和后续 FFN 可直接复用

## 怎么验证

- 输出形状不变
- 归一化后数值分布合理
- 激活函数输出符合预期范围

如果输入整体偏移后，LayerNorm 输出仍能回到稳定范围，这通常说明归一化逻辑没有跑偏。

一个最小的 LayerNorm 伪代码可以写成：

```cpp
for (each token) {
    mean = average(x);
    var = average((x - mean)^2);
    y = (x - mean) / sqrt(var + eps) * gamma + beta;
}
```

激活函数则可以单独作为纯函数实现，便于在 FFN 中复用。

## 常见问题

- 归一化维度选错
- 方差计算不稳定
- 激活函数实现和公式不一致
