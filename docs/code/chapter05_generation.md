# chapter05_generation 代码索引

## 阅读顺序

1. 先看 [chapter05 导读](../chapter05_generation/intro)
2. 再看 `src/request.cpp`、`src/runtime.cpp`
3. 然后看 `src/kv_cache.cpp`、`src/scheduler.cpp`
4. 再看 `src/sampler.cpp`、`src/transformer.cpp`
5. 最后看测试和脚本

## 运行时骨架

<h3 id="request-cpp">src/request.cpp</h3>

- `code/chapter05_generation/src/request.cpp`

<h3 id="runtime-cpp">src/runtime.cpp</h3>

- `code/chapter05_generation/src/runtime.cpp`

<h3 id="kv-cache-h">src/kv_cache.h</h3>

- `code/chapter05_generation/src/kv_cache.h`

<h3 id="kv-cache-cpp">src/kv_cache.cpp</h3>

- `code/chapter05_generation/src/kv_cache.cpp`

<h3 id="scheduler-cpp">src/scheduler.cpp</h3>

- `code/chapter05_generation/src/scheduler.cpp`

<h3 id="sampler-cpp">src/sampler.cpp</h3>

- `code/chapter05_generation/src/sampler.cpp`

<h3 id="transformer-cpp">src/transformer.cpp</h3>

- `code/chapter05_generation/src/transformer.cpp`

<h3 id="decoder-layer-cpp">src/decoder_layer.cpp</h3>

- `code/chapter05_generation/src/decoder_layer.cpp`

## 示例与测试

<h3 id="main-generation-cpp">examples/main_generation.cpp</h3>

- `code/chapter05_generation/examples/main_generation.cpp`

<h3 id="main-streaming-cpp">examples/main_streaming.cpp</h3>

- `code/chapter05_generation/examples/main_streaming.cpp`

<h3 id="run-sh">scripts/run.sh</h3>

- `code/chapter05_generation/scripts/run.sh`

<h3 id="runtime-smoke-test-cpp">tests/runtime_smoke_test.cpp</h3>

- `code/chapter05_generation/tests/runtime_smoke_test.cpp`

<h3 id="kv-cache-test-cpp">tests/kv_cache_test.cpp</h3>

- `code/chapter05_generation/tests/kv_cache_test.cpp`

<h3 id="sampler-test-cpp">tests/sampler_test.cpp</h3>

- `code/chapter05_generation/tests/sampler_test.cpp`

<h3 id="chapter05-scheduler-test-cpp">tests/chapter05_scheduler_test.cpp</h3>

- `code/chapter05_generation/tests/chapter05_scheduler_test.cpp`

<h3 id="chapter05-generation-test-cpp">tests/chapter05_generation_test.cpp</h3>

- `code/chapter05_generation/tests/chapter05_generation_test.cpp`

## 对应文档

- [chapter05 导读](../chapter05_generation/intro)
- [chapter05 小结](../chapter05_generation/summary)

## 说明

这一章的代码重点是把 `forward` 变成可验证的 generation runtime。读者应当先跟着阅读顺序看核心文件，再用测试和脚本收口，而不是只扫目录列表。
