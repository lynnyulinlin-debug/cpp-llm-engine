# 05.8 Generation Runtime 与 vLLM / SGLang 对照

本章已经把 `prefill -> decode -> sample -> output` 做成可观察状态机。成熟 serving runtime 在这条链上增加动态批处理、分页缓存、抢占和分布式执行。

## 先按问题维度对照

Continuous Batching、请求排序、KV 物理管理和前缀复用会在一次 schedule step 中协作，但属于四类不同决策。先分开观察，再跟踪它们怎样交换 metadata。

| 问题维度 | 本教程 | vLLM 坐标 | Mini-SGLang / SGLang 坐标 |
| --- | --- | --- | --- |
| 本 step 推进谁 | `StepBatchPlanner` 混合 prefill/decode | iteration-level scheduler | Prefill/Decode manager、overlap scheduler |
| waiting 请求怎样排序 | admission 顺序、最少进度优先 | scheduler policy 与 priority | FCFS、LPM 等 schedule policy |
| K/V 放在哪里 | `KvBlockPool`、`BlockTable` 只管理 block ownership | KV cache manager、block pool | token/page pool 与 cache manager |
| 已算前缀怎样复用 | 尚未跨请求复用 | prefix caching | Radix Cache / RadixAttention |
| 冷数据放在哪里 | 单级 CPU 教学 pool | cache/offload 扩展 | HiCache 的 GPU、主机和外部存储层级 |
| CPU 调度怎样隐藏 | 同步 step | 异步执行扩展 | overlap scheduling |

## 源码入口

先跟踪一次 schedule step：选择请求、分配 cache、执行模型、更新请求状态，再回到下一轮。

| 框架 | 官方入口 | 带着什么问题读 |
| --- | --- | --- |
| vLLM | [`scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) | waiting/running 请求怎样组成一次执行计划 |
| vLLM | [`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) | cache block 何时分配、提交和释放 |
| vLLM | [`engine/core.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py) | scheduler、executor 与输出更新怎样闭环 |
| Mini-SGLang | [`scheduler.py`](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/scheduler/scheduler.py) | normal/overlap loop 怎样交换 batch 和执行结果 |
| SGLang | [`scheduler.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py) | token/cache budget 怎样进入生产调度循环 |
| SGLang | [`radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) | prefix match、锁定、插入和淘汰怎样改变 cache ownership |

## 教程保留与省略

| 已实现 | 暂不实现 | 后续位置 |
| --- | --- | --- |
| 单请求和多请求 prefill/decode 状态 | 抢占、优先级和 speculative decode | Chapter09/服务扩展 |
| FIFO/Round Robin、最少进度优先与 token budget | priority、LPM、抢占和 overlap scheduling | Chapter09 / SGLang 对照 |
| CPU block pool 与 block table | device K/V payload 和 paged attention kernel | GPU backend |
| callback streaming | HTTP/SSE、背压和断连 | Chapter09 |

## 源码阅读问题

1. scheduler 输出的是请求列表、token 数，还是完整执行 metadata？
2. cache manager 和 scheduler 谁拥有 block，失败时由谁回收？
3. chunked prefill 加入后，公平性和 TTFT 为什么可能同时变化？
4. LPM 改变的是等待队列顺序，还是 Radix Cache 的 prefix match 算法？

项目定位继续阅读 [`vLLM`](../reference/ecosystem/vllm.md)、[`Mini-SGLang`](../reference/ecosystem/mini_sglang.md) 和 [`SGLang`](../reference/ecosystem/sglang.md) 项目卡。
