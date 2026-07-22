# chapter06_quantization 代码索引

## 阅读顺序

1. 先看 [chapter06 导读](../chapter06_quantization/intro)
2. 再看 `src/tensor.cpp` 和 `src/dequantize_op.cpp`
3. 然后看 `src/linear_layer.cpp` 和 `src/quantized_linear.cpp`
4. 最后看测试和脚本

## 量化路径

- <h3 id="tensor-cpp">src/tensor.cpp</h3>
- `code/chapter06_quantization/src/tensor.cpp`
- <h3 id="dequantize-op-cpp">src/dequantize_op.cpp</h3>
- `code/chapter06_quantization/src/dequantize_op.cpp`
- <h3 id="linear-layer-cpp">src/linear_layer.cpp</h3>
- `code/chapter06_quantization/src/linear_layer.cpp`
- <h3 id="quantized-linear-cpp">src/quantized_linear.cpp</h3>
- `code/chapter06_quantization/src/quantized_linear.cpp`

## 示例与测试

- <h3 id="main-quantization-cpp">examples/main_quantization.cpp</h3>
- `code/chapter06_quantization/examples/main_quantization.cpp`
- <h3 id="run-sh">scripts/run.sh</h3>
- `code/chapter06_quantization/scripts/run.sh`
- <h3 id="chapter06-quantization-test-cpp">tests/chapter06_quantization_test.cpp</h3>
- `code/chapter06_quantization/tests/chapter06_quantization_test.cpp`

## 对应文档

- [chapter06 导读](../chapter06_quantization/intro)
- [chapter06 小结](../chapter06_quantization/summary)

## 说明

这一章的代码重点是让量化和反量化进入可验证的推理路径。读者应当先跟着阅读顺序看核心文件，再回到测试和脚本收口，而不是只看文件列表。
