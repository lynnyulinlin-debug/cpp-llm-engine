# 05.8 Generation Runtime 框架对照

Chapter05 把 `prefill -> decode -> sample -> output` 做成可观察状态机。成熟 Runtime 在相同链路上增加动态批处理、分页缓存、前缀复用、抢占和异步执行。

## 问题维度

Continuous Batching、请求排序、KV 物理管理和前缀复用会在一次 schedule step 中协作，但分别回答不同问题。

| 问题 | 本教程 | vLLM | Mini-SGLang / SGLang |
| --- | --- | --- | --- |
| 本 step 推进谁 | `StepBatchPlanner` 混合 prefill/decode | iteration-level scheduler | Prefill/Decode manager、overlap scheduler |
| waiting 请求怎样排序 | admission 顺序、最少进度优先 | scheduler policy 与 priority | FCFS、LPM 等策略 |
| K/V 放在哪里 | `KvBlockPool`、`BlockTable` 管理 ownership | KV cache manager、block pool | token/page pool 与 cache manager |
| 已算前缀怎样复用 | 请求级 K/V | prefix caching | Radix Cache / RadixAttention |
| CPU 调度怎样隐藏 | 同步 step | 异步执行扩展 | overlap scheduling |

## 教学实现与生产实现

教学 Runtime 用确定性 CPU step 验证状态迁移和资源回收；生产系统继续处理设备页、跨请求复用和异步并发。

| 本章可验证对象 | 成熟系统继续增加 | 后续入口 |
| --- | --- | --- |
| 单请求和多请求 prefill/decode | 抢占、优先级和 speculative decode | Chapter09 |
| 最少进度优先与 token budget | priority、LPM 和 overlap scheduling | Chapter09、SGLang 项目卡 |
| CPU block pool 与 block table | device K/V payload、paged attention | CUDA backend |
| callback streaming | HTTP/SSE、背压和断连 | Chapter09 |

## 源码阅读问题

从项目卡进入源码后，跟踪一次 `选择请求 -> 分配 cache -> 执行模型 -> 更新状态`：

1. Scheduler 输出请求列表、token 数，还是完整执行 metadata？
2. Cache manager 和 Scheduler 分别持有哪些 block 状态，失败时谁负责回收？
3. Chunked Prefill 加入后，公平性和 TTFT 为什么可能同时变化？
4. LPM 改变 waiting 顺序时，怎样使用 Radix Cache 的 prefix match 结果？

| 项目卡 | 在本章中的用途 |
| --- | --- |
| [vLLM](../reference/ecosystem/vllm.md) | Continuous Batching、Block KV 和 EngineCore |
| [Mini-SGLang](../reference/ecosystem/mini_sglang.md) | 紧凑的 Prefill/Decode 与 overlap 调度骨架 |
| [SGLang](../reference/ecosystem/sglang.md) | LPM、Radix Cache 和多级缓存扩展 |
