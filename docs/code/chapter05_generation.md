# chapter05_generation 代码导航

这一页用于定位 generation runtime 的请求、调度、KV cache、采样和端到端测试。

## 两部分入口

- [A. Generation Core](../chapter05_generation/part_a_generation_core.md)
- [B. Serving Preview](../chapter05_generation/part_b_serving_preview.md)

## 阅读顺序

1. 先看 `request.cpp` 和 `runtime.cpp`
2. 再看 `kv_cache.cpp`、`scheduler.cpp`、`sampler.cpp`
3. 最后看测试和脚本

## 运行时骨架

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter05_generation/include/chapter05_generation/request.h"><code>request.h</code></a> | 请求状态接口 |
| <a href="/_code/chapter05_generation/src/request.cpp"><code>src/request.cpp</code></a> | 请求生命周期 |
| <a href="/_code/chapter05_generation/include/chapter05_generation/runtime.h"><code>runtime.h</code></a> | Runtime 接口 |
| <a href="/_code/chapter05_generation/src/runtime.cpp"><code>src/runtime.cpp</code></a> | 生成 runtime |
| <a href="/_code/chapter05_generation/src/kv_cache.cpp"><code>src/kv_cache.cpp</code></a> | KV Cache |
| <a href="/_code/chapter05_generation/src/scheduler.cpp"><code>src/scheduler.cpp</code></a> | Scheduler |
| <a href="/_code/chapter05_generation/src/sampler.cpp"><code>src/sampler.cpp</code></a> | Sampler |
| <a href="/_code/chapter05_generation/src/transformer.cpp"><code>src/transformer.cpp</code></a> | Transformer 组装 |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter05_generation/examples/main_generation.cpp"><code>examples/main_generation.cpp</code></a> | 生成示例 |
| <a href="/_code/chapter05_generation/examples/main_streaming.cpp"><code>examples/main_streaming.cpp</code></a> | 流式示例 |
| <a href="/_code/chapter05_generation/tests/runtime_smoke_test.cpp"><code>tests/runtime_smoke_test.cpp</code></a> | runtime smoke test |
| <a href="/_code/chapter05_generation/tests/chapter05_generation_test.cpp"><code>tests/chapter05_generation_test.cpp</code></a> | 端到端测试 |
| <a href="/_code/chapter05_generation/scripts/run.sh"><code>scripts/run.sh</code></a> | 运行脚本 |

## 对应文档

- [chapter05 导读](../chapter05_generation/intro.md)

## 说明

代码索引页只做补充导航。正文页的“源码直达”应直接链接到具体文件，不应只跳到本页或章节代码目录。
