# Chapter 05：Generation Runtime 生成运行时

本目录把 Decoder 结构连接成 `prefill -> decode -> sampling -> output` 的最小 Generation Runtime，并显式管理 Request、KV Cache、Scheduler、token-budget step 和 streaming。服务 API、限流和资源池由 Chapter09 承接。完整学习说明见 [`source/chapter05_generation`](../../source/chapter05_generation/intro.md)。

## 本章资产总览

| 部分 | 职责 |
| --- | --- |
| A. Generation Core | `prefill -> decode -> sampling -> output`、Request、KV Cache、基础 Scheduler 和 Streaming |
| B. Serving Preview | token-level Continuous Batching、chunked prefill、公平性、物理 block ID 和 BlockTable；队列与服务接入由 Chapter09 承接 |

| 资产 | 作用 |
| --- | --- |
| `src/request.cpp` | Request 状态、取消和超时 |
| `src/kv_cache.cpp` | 历史 K/V 的分配和增长 |
| `src/embedding.cpp`、`position_encoding.cpp` | 输入 embedding 和位置编码 |
| `src/transformer.cpp`、`decoder_layer.cpp` | prefill/decode 的模型结构入口 |
| `src/sampler.cpp` | logits 到 token 的采样 |
| `src/scheduler.cpp` | 多请求 round-robin 推进 |
| `include/chapter05_generation/model_executor.h` | 每请求完整/分块 prefill 与 decode 执行器契约 |
| `src/serving_preview.cpp` | token/block budget、最少 prefill 进度优先、`KvBlockPool` 和物理 BlockTable |
| `src/runtime.cpp` | 状态、cache、sampler 的运行时编排 |
| `tests/` | generation、scheduler、KV Cache、sampler、Serving Preview 和 smoke 测试 |
| `examples/main_generation.cpp` | 普通生成 demo |
| `examples/main_streaming.cpp` | streaming callback demo |
| `scripts/run.sh` | 隔离构建、测试和运行两个 demo |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter05_generation/scripts/run.sh
```

验证重点是请求状态、完整/分块 prefill、step token/block budget、公平推进、block 释放复用和 streaming 输出。`ContinuousBatchScheduler::run_step()` 在模型执行前事务性预留 block，未推进时撤销预留；中间 chunk 只追加 Context，final chunk 才产生 logits。`KvBlockPool` 管理物理 block ID 和容量，真实分层 K/V tensor 仍由每请求 executor Context 持有。
