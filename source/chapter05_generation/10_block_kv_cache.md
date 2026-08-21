# 05.11 Block KV Cache 阅读入口

`chapter05` 已经实现了最小 KV Cache，但 vLLM / nano-vLLM 的关键工程差异在于：KV Cache 通常不会按“每个请求一整段连续内存”来理解，而会被切成 block，再通过 block table 把逻辑 token 位置映射到物理缓存块。

## 阅读入口

- 前置阅读：[`05.4 KV Cache`](./04_kv_cache.md)、[`05.10 Continuous Batching 入门`](./09_continuous_batching.md)
- 源码直达：<a href="../_code/chapter05_generation/src/kv_cache.cpp"><code>kv_cache.cpp</code></a>、<a href="../_code/chapter05_generation/src/request.cpp"><code>request.cpp</code></a>
- 相关阅读：[`05.8 vLLM 对照`](./07_vllm_comparison.md)、[`推理引擎运行时系统`](../reference/runtime_systems.md)、[`vLLM 生态对照`](../reference/ecosystem/vllm.md)

## 为什么要 block

连续缓存最容易理解，但它会带来两个问题：

- 请求长度不同，连续分配容易产生浪费或搬迁
- Continuous Batching 需要请求动态进出，cache 必须能被更灵活地分配和释放

Block KV Cache 的思想是把 cache 分成固定大小的块。请求看到的是逻辑 token 序列，runtime 内部用 block table 找到每段 token 对应的物理块。

## 先看懂三个对象

| 对象 | 作用 |
| --- | --- |
| block | 固定大小的 KV cache 存储单元 |
| block table | 把请求的逻辑 token 位置映射到物理 block |
| slot | decode 新 token 写入 cache 的具体位置 |

有了这三个对象，KV Cache 就从“一个请求一个 vector”变成了“请求持有一张映射表，底层 cache 统一管理物理块”。

## 和 PagedAttention 的关系

PagedAttention 可以先按内存管理思想理解：它让 attention 在读取历史 K/V 时，通过 block table 找到实际缓存位置。这样请求不需要占用一整段连续 cache，也更适合 continuous batching 里的动态进出。

本教程当前不要求实现完整 PagedAttention kernel。入门阶段更重要的是先看懂这条边界：

```text
request logical tokens -> block table -> physical KV blocks -> attention reads K/V
```

这条链路是阅读 nano-vLLM 或 vLLM 的 block manager、cache engine、attention metadata 的前置。它将“调度请求”和“寻址 KV cache”分成两个独立问题。
