# 06.1 Quantization Basics 量化基础


## 这一页要解决什么

理解量化的基本原理、数值映射和误差来源，为后续 INT8 推理路径做准备。

## 阅读入口

- 前置阅读：[`chapter06 导读`](/chapter06_quantization/intro)、[`chapter05_generation`](/chapter05_generation/intro)
- 源码直达：[`chapter06_quantization 代码入口`](/code/chapter06_quantization)
- 相关阅读：[`06.2 Extend Tensor`](/chapter06_quantization/02_extend_tensor)、[`06.3 Load Quantized Weights`](/chapter06_quantization/03_load_quantized_weights)

## 代码直达

- [`src/tensor.cpp`](/code/chapter06_quantization#tensor-cpp)
- [`src/dequantize_op.cpp`](/code/chapter06_quantization#dequantize-op-cpp)
- [`src/linear_layer.cpp`](/code/chapter06_quantization#linear-layer-cpp)
- [`tests/chapter06_quantization_test.cpp`](/code/chapter06_quantization#chapter06-quantization-test-cpp)

## 读完后应能做什么

- 理解为什么需要量化
- 理解同步和非同步量化的基本概念
- 理解 INT8、INT16 等数值表示之间的转换
- 理解量化误差为什么会出现

## 为什么这一步重要

量化不是简单地把位宽变小，而是要在存储、带宽、速度和精度之间做权衡。先理解基础原理，后面的实现和验证才不会变成“只会调参数”。

## 这一页的核心内容

- 量化的目的
- INT8 / INT16 映射关系
- scale / zero point / rounding
- 量化误差和饱和

在工程上，量化不是“把 float 改成 int”这么简单，而是要定义数值范围、映射规则和误差承受区间。这个定义不清楚，后面的 loader 和算子都没法稳定实现。

最小版本通常先从对称量化或最简单的 scale-based 映射开始。

同步量化（Symmetric Quantization）指零点固定为 0，数值通常映射到 `[-scale * 127, scale * 127]` 这样的对称区间；非同步量化（Asymmetric Quantization）则会引入 `zero_point`，用来更充分地利用整数范围，尤其适合分布不对称的激活值。本教程先从对称量化开始，因为它实现更简单，也更容易和 GEMM / dequantize 路径对齐。

这里还有一个很重要的点：不同层的敏感程度不同。某些层对量化误差更敏感，某些层相对宽松。教程里不需要一次讲到所有复杂策略，但最好让读者意识到“量化策略”本身也是工程选择。

最基础的 scale-based 映射伪代码可以写成：

```cpp
int8_t q = round(x / scale);
float x_hat = q * scale;
```

这段公式虽然简单，但已经足以表达量化和反量化的核心关系。

## 实现思路

- 先写清楚数值转换公式
- 再讨论不同量化策略
- 最后再进入工程实现

## 怎么验证

- 能说明浮点到整数的转换过程
- 能解释为什么会有误差
- 能区分量化前后表示的差异

能够给出一个具体浮点值，手算出它被量化后的整数，再手算反量化结果，这一步就算真正理解了。

## 常见问题

- 把量化当成简单的类型转换
- 忽略误差和饱和
- 没有区分权重量化和激活量化
