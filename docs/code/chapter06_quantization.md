# chapter06_quantization 代码导航

这一页用于定位 INT8 权重、反量化、量化 Linear 和精度验证。

## 阅读顺序

1. 先看 `tensor.cpp` 和 `dequantize_op.cpp`
2. 再看 `linear_layer.cpp` 和 `quantized_linear.cpp`
3. 最后看测试

## 量化路径

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter06_quantization/include/chapter06_quantization/tensor.h"><code>tensor.h</code></a> | 量化 Tensor 接口 |
| <a href="/_code/chapter06_quantization/src/tensor.cpp"><code>src/tensor.cpp</code></a> | 量化 Tensor 实现 |
| <a href="/_code/chapter06_quantization/src/dequantize_op.cpp"><code>src/dequantize_op.cpp</code></a> | 反量化算子 |
| <a href="/_code/chapter06_quantization/src/linear_layer.cpp"><code>src/linear_layer.cpp</code></a> | FP baseline Linear |
| <a href="/_code/chapter06_quantization/src/quantized_linear.cpp"><code>src/quantized_linear.cpp</code></a> | 量化 Linear |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter06_quantization/examples/main_quantization.cpp"><code>examples/main_quantization.cpp</code></a> | 量化示例 |
| <a href="/_code/chapter06_quantization/tests/chapter06_quantization_test.cpp"><code>tests/chapter06_quantization_test.cpp</code></a> | 章节测试 |
| <a href="/_code/chapter06_quantization/scripts/run.sh"><code>scripts/run.sh</code></a> | 运行脚本 |

## 对应文档

- [chapter06 导读](../chapter06_quantization/intro.md)
- [chapter06 小结](../chapter06_quantization/summary.md)

## 说明

代码索引页只做补充导航。正文页的“源码直达”应直接链接到具体文件，不应只跳到本页或章节代码目录。
