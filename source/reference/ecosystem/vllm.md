# vLLM 高吞吐 Serving

## 解决什么问题

`vLLM` 主要解决高吞吐的 LLM serving 问题。它特别适合回答这样一个问题：如何在控制 KV cache 的前提下，高效管理很多并发请求，并把吞吐、延迟和缓存管理放进同一套 runtime 里。

## 工程策略

- 动态批处理请求
- 强力管理 KV cache
- 通过 scheduler 调整吞吐和延迟之间的平衡

## 核心设计模式

- `PagedAttention`：把 KV cache 按块组织，减少碎片和搬运成本
- `Continuous Batching`：让 batch 随请求状态动态变化，而不是一次固定到底
- `Prefix Caching`：复用共享前缀的计算结果，降低重复开销

这些模式的共同点是：它们都在解决“算得对之后，怎么让 runtime 更高效”。

## 场景决策

- 如果你在做单机本地推理，先看 `llama.cpp`
- 如果你在做模型前向和 runtime 的主线构建，先看本教程的 `chapter05_generation`
- 如果你在做高吞吐 serving，重点看 `vLLM`
- 如果你在做应用编排和结构化 serving，再看 `SGLang`

## 什么时候选它

如果你的问题是服务端吞吐和请求调度，就优先看 `vLLM`：

- 并发请求很多
- 需要在负载下复用 KV cache
- 需要权衡延迟和吞吐
- 批处理行为会随着运行时变化

## 对应教程章节

`vLLM` 主要对应教程里的这些章节：

- `chapter05_generation`
- `chapter07_benchmark`
- `chapter04_attention` 作为 decoder 结构基础

在本教程里，`vLLM` 不是一个需要“先学完再回来”的主线章节，而是一个在你已经掌握 `chapter05_generation` 和 `chapter07_benchmark` 之后，用来对照 serving 设计取舍的参考点。

## 迁移入口

- `chapter05_generation`：先把朴素 generation、调度器、KV 管理雏形跑通，再看 `vLLM` 的动态批处理和 cache 组织方式
- `chapter07_benchmark`：用 benchmark 和对比结果判断哪些 serving 思想值得迁移到自己的 runtime
- `docs/reference/ecosystem/README.md`：先按场景选择框架，再决定是否深入 `vLLM`

## 未实现范围

本教程不追求 `vLLM` 的 1:1 复刻，当前不包含：

- 完整的 `prefix caching`
- 更复杂的 `chunked prefill`
- `speculative decoding`
- 更完整的 page/block 管理策略
- 更复杂的抢占与恢复机制

这些内容如果后续要补，应该作为后续进阶专题，而不是打散当前主线。

## 怎么看这份参考

- 把 `vLLM` 当作 serving 设计标杆，而不是源码模板
- 重点比较 request scheduling、continuous batching、KV cache 和 streaming
- 先用本教程的主线理解 runtime，再回来看 `vLLM` 的设计取舍
