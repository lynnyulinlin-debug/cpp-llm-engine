# 09.5 Serving 框架对照：ServiceCore 与生产服务之间还差什么

本章把请求接入、调度、执行、流式输出和资源释放放进一个 C++ ServiceCore。成熟框架会继续把这些职责拆到 API server、engine core、worker 和 cache manager 中。

## 职责映射

| 教程对象 | vLLM | SGLang | `llama.cpp` server |
| --- | --- | --- | --- |
| HTTP/SSE transport | OpenAI-compatible API server | HTTP server / tokenizer manager | server endpoint 与 slot |
| `ServiceCore::submit` | engine request admission | scheduler request admission | request/slot 分配 |
| waiting/running 队列 | V1 scheduler | SRT scheduler | slot 状态推进 |
| `ModelExecutorFactory` | executor / worker / model runner | model worker | server context + model context |
| block ownership 与回收 | KV cache manager | radix/prefix cache | context KV cache |
| metrics 与 benchmark JSON | serving metrics | scheduler/server metrics | timings 与 server metrics |

## 调度与缓存维度

服务对象映射之后，还要把调度与缓存拆成独立维度。Continuous Batching 决定本 step 的执行集合，LPM 决定 waiting 请求顺序，Radix Cache 决定共享前缀如何索引和持有 K/V，HiCache 决定前缀数据可以存在哪一级。

| 维度 | 本教程当前对象 | 成熟系统继续增加 |
| --- | --- | --- |
| Batch composition | `dispatch_step()`、token/block budget | prefill/decode 混排、抢占和异步执行 |
| Queue policy | admission order、round-robin/最少进度 | priority、deadline、LPM |
| Physical KV ownership | `KvBlockPool`、请求级 `BlockTable` | device page pool、复制、swap/offload |
| Prefix reuse | 请求之间不共享 K/V | hash prefix cache 或 Radix Cache |
| Storage hierarchy | 单级进程内资源 | GPU/CPU/外部存储和预取 |

## 源码入口

先沿一条请求从 API 到 engine、再到终态回收，不要同时阅读所有服务特性。

| 框架 | 官方入口 | 带着什么问题读 |
| --- | --- | --- |
| vLLM | [`engine/core.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py)、[`scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) | API 进程与 engine core 如何通信，请求怎样被调度 |
| vLLM | [`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) | cache block 何时分配、提交和释放 |
| Mini-SGLang | [`scheduler.py`](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/scheduler/scheduler.py) | normal/overlap loop 怎样保持可读的请求推进路径 |
| SGLang | [`scheduler.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)、[`radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) | schedule policy 与 prefix cache 怎样交换命中和 ownership 信息 |
| `llama.cpp` | [`tools/server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | 本地 C++ 模型上下文怎样接到 HTTP 服务 |

## 教程保留与省略

本章优先保证服务语义可测试，因此 transport 可以替换，ServiceCore、metrics 和 streaming contract 保持稳定。

| 已实现 | 暂不实现 | 省略原因 |
| --- | --- | --- |
| admission、waiting/running、cancel、timeout | 多租户优先级、抢占和全局路由 | 先验证单实例生命周期 |
| 同步完成和 SSE token event | 完整 OpenAI schema、鉴权、TLS | transport 不应主导 engine 设计 |
| 每请求 Executor/Context 与统一回收 | 多进程 worker、分布式 executor | 先固定资源所有权 |
| queue wait、TTFT、completion 聚合 | histogram、P99、分布式 trace | 先保证指标来源正确 |
| token/block-budget dispatch | GPU 并行 batch executor | CPU 主线先证明调度行为 |

## 源码阅读问题

1. transport 断开、取消或超时时，终态由哪一层写入并触发回收？
2. scheduler 只选择请求，还是也拥有 KV block 和模型执行资源？
3. API server、engine core 和 worker 拆进不同进程后，哪些契约必须保持不变？
4. 前缀命中、waiting 排序和 KV 存储层级分别由哪个对象决定？

项目定位继续阅读 [`vLLM`](../reference/ecosystem/vllm.md)、[`Mini-SGLang`](../reference/ecosystem/mini_sglang.md) 和 [`SGLang`](../reference/ecosystem/sglang.md) 项目卡。
