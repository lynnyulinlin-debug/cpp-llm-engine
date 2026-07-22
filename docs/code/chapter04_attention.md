# chapter04_attention 代码索引

## 阅读顺序

1. 先看 [chapter04 导读](../chapter04_attention/intro)
2. 再看 `src/attention.cpp` 和 `src/feed_forward.cpp`
3. 然后看 `src/decoder_layer.cpp`
4. 最后用测试和脚本收口

## 核心文件

- <h3 id="attention-cpp">src/attention.cpp</h3>
- `code/chapter04_attention/src/attention.cpp`
- <h3 id="feed-forward-cpp">src/feed_forward.cpp</h3>
- `code/chapter04_attention/src/feed_forward.cpp`
- <h3 id="decoder-layer-cpp">src/decoder_layer.cpp</h3>
- `code/chapter04_attention/src/decoder_layer.cpp`
- <h3 id="layer-norm-cpp">src/layer_norm.cpp</h3>
- `code/chapter04_attention/src/layer_norm.cpp`
- <h3 id="linear-layer-cpp">src/linear_layer.cpp</h3>
- `code/chapter04_attention/src/linear_layer.cpp`
- <h3 id="tensor-cpp">src/tensor.cpp</h3>
- `code/chapter04_attention/src/tensor.cpp`

## 示例与测试

- <h3 id="main-attention-cpp">examples/main_attention.cpp</h3>
- `code/chapter04_attention/examples/main_attention.cpp`
- <h3 id="run-sh">scripts/run.sh</h3>
- `code/chapter04_attention/scripts/run.sh`
- <h3 id="chapter04-attention-test-cpp">tests/chapter04_attention_test.cpp</h3>
- `code/chapter04_attention/tests/chapter04_attention_test.cpp`

## 对应文档

- [chapter04 导读](../chapter04_attention/intro)
- [chapter04 小结](../chapter04_attention/summary)

## 说明

这一章的代码重点是把算子组合成可验证的单层 decoder。读者应当先理解章节主线，再顺着核心文件和测试往下看，而不是从目录树里盲找文件。
