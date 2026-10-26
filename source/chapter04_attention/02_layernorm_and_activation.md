# 04.2 LayerNorm and Activation LayerNorm 与激活函数


## 组合 LayerNorm 和 Activation

实现教学 Decoder 使用的 LayerNorm 和 ReLU，并理解它们与 TinyLlama 的 RMSNorm 和 SwiGLU 的差异。

读完后应能做到：

- 理解 LayerNorm 的作用
- 理解激活函数在非线性表达中的意义
- 区分已实现的 ReLU 与模型变体中的 SiLU、GELU

## 阅读入口

- 前置阅读：[`chapter04 导读`](./intro.md)、[`04.1 Multi-head Attention`](./01_multihead_attention.md)
- 源码直达：<a href="../_code/chapter04_attention/src/layer_norm.cpp"><code>layer_norm.cpp</code></a>、<a href="../_code/chapter04_attention/tests/chapter04_attention_test.cpp"><code>chapter04_attention_test.cpp</code></a>
- 相关阅读：[`04.3 Feed Forward Network`](./03_feedforward.md)、[`04.4 DecoderLayer`](./04_decoder_layer.md)

## 核心实现

LayerNorm 稳定训练和推理中的数值分布，激活函数引入非线性表达能力。它们虽然不是最“显眼”的模块，但对 Transformer 的稳定性很关键。

仓库教学层实现以下内容：

- LayerNorm 的均值和方差归一化
- gamma 和 beta 参数位置；教学构造函数固定初始化为 1 和 0
- `apply_activation(..., ActivationKind::Relu)`

SiLU 和 GELU 用于模型结构对照，Chapter04 源码没有实现这两种激活。真实 TinyLlama 使用 RMSNorm 与 SwiGLU，其实现位于真实 CPU Decoder，不能用教学 `LayerNorm` 和 ReLU 代替后声称数值等价。

工程上建议把 LayerNorm 的参数、统计量计算和输出变换分开写清楚，这样后续复用更简单。

LayerNorm 的归一化维度：

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

SiLU 和 GELU：

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

教学源码把 ReLU 做成独立纯函数供 FFN 复用。若扩展 GELU，应明确近似公式和误差测试；若实现 TinyLlama 路径，则需要 gate/up 两条投影和 SiLU 乘门，而不是只替换一个枚举值。

推理中的一致性：

LayerNorm 在训练和推理里的行为是一致的：

- 训练：按当前输入计算均值和方差
- 推理：仍然按当前输入计算均值和方差

所以它不需要像 BatchNorm 那样维护 running mean / running var。对本教程来说，这意味着同一份 LayerNorm 实现可以直接用于推理路径。

实现建议：

- 先实现基础公式
- 再关注数值稳定性
- 保持接口和后续 FFN 可直接复用

## 验证与排查

验证时重点看：

- 输出形状不变
- 归一化后数值分布合理
- ReLU 将负值截断为 0，并保留正值

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

常见问题：

- 归一化维度选错
- 方差计算不稳定
- 激活函数实现和公式不一致
