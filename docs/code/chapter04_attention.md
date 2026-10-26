# chapter04_attention 代码导航

这一页用于定位 attention、FFN、decoder layer 和对应测试。

## 阅读顺序

1. 先看 `attention.cpp`
2. 再看 `feed_forward.cpp` 和 `decoder_layer.cpp`
3. 最后用测试收口

## 模型结构

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter04_attention/src/attention.cpp"><code>src/attention.cpp</code></a> | 多头注意力 |
| <a href="/_code/chapter04_attention/src/feed_forward.cpp"><code>src/feed_forward.cpp</code></a> | FFN |
| <a href="/_code/chapter04_attention/src/decoder_layer.cpp"><code>src/decoder_layer.cpp</code></a> | DecoderLayer 组合 |
| <a href="/_code/chapter04_attention/src/layer_norm.cpp"><code>src/layer_norm.cpp</code></a> | LayerNorm |
| <a href="/_code/chapter04_attention/src/linear_layer.cpp"><code>src/linear_layer.cpp</code></a> | Linear |
| <a href="/_code/chapter04_attention/src/tensor.cpp"><code>src/tensor.cpp</code></a> | 本章 Tensor |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter04_attention/examples/main_attention.cpp"><code>examples/main_attention.cpp</code></a> | 示例入口 |
| <a href="/_code/chapter04_attention/tests/chapter04_attention_test.cpp"><code>tests/chapter04_attention_test.cpp</code></a> | 章节测试 |
| <a href="/_code/chapter04_attention/scripts/run.sh"><code>scripts/run.sh</code></a> | 运行脚本 |

## 对应文档

- [chapter04 导读](../chapter04_attention/intro.md)
- [chapter04 导读](../chapter04_attention/intro.md)

## 说明

代码索引页只做补充导航。正文页的“源码直达”应直接链接到具体文件，不应只跳到本页或章节代码目录。
