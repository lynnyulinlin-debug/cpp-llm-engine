# chapter05_generation 代码索引

## 运行时骨架

- `code/chapter05_generation/src/request.cpp`
- `code/chapter05_generation/src/runtime.cpp`
- `code/chapter05_generation/src/kv_cache.cpp`
- `code/chapter05_generation/src/scheduler.cpp`
- `code/chapter05_generation/src/sampler.cpp`
- `code/chapter05_generation/src/transformer.cpp`
- `code/chapter05_generation/src/decoder_layer.cpp`

## 示例与测试

- `code/chapter05_generation/examples/main_generation.cpp`
- `code/chapter05_generation/examples/main_streaming.cpp`
- `code/chapter05_generation/scripts/run.sh`
- `code/chapter05_generation/tests/runtime_smoke_test.cpp`
- `code/chapter05_generation/tests/kv_cache_test.cpp`
- `code/chapter05_generation/tests/sampler_test.cpp`
- `code/chapter05_generation/tests/chapter05_scheduler_test.cpp`
- `code/chapter05_generation/tests/chapter05_generation_test.cpp`

## 对应文档

- [chapter05 导读](../chapter05_generation/intro)
- [chapter05 小结](../chapter05_generation/summary)

## 说明

这一章的代码重点是把 `forward` 变成可验证的 generation runtime。
