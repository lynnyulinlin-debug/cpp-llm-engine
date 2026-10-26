# 09.1 C++ Service Core

## 先运行最小闭环

从仓库根目录运行：

```bash
bash code/chapter09_serving/scripts/run.sh
```

`ServingCore(3, 2)` 最多容纳 3 个 inflight request，每轮最多向 Runtime 提交 2 个请求。demo 先完成 2 个请求，释放资源后再处理剩余请求。

源码按“接口、状态推进、断言”顺序阅读：

- <a href="/_code/chapter09_serving/include/chapter09_serving/serving_core.h"><code>serving_core.h</code></a>
- <a href="/_code/chapter09_serving/src/serving_core.cpp"><code>serving_core.cpp</code></a>
- <a href="/_code/chapter09_serving/tests/chapter09_serving_test.cpp"><code>chapter09_serving_test.cpp</code></a>

## 请求生命周期

`ServingCore` 保存请求引用、请求 ID、Sampler、可选的 `RequestModelExecutor` 和队列位置。Chapter05 `Request` 保存生成状态，Runtime 调用 executor 完成模型计算。`dispatch_step(runtime, max_batch_tokens)` 按 token budget 追加 prompt 或执行 decode；配置 `KvBlockPool` 后，还会在调用 executor 前预留 block。

未提供 executor 时，测试使用教学 Transformer；提供 executor 时，round-robin 分别调用每个请求的 prefill/decode。`RealLlamaExecutorFactory` 只加载一份 `RealLlamaCpuModel`，每次 admission 创建独立 `RealLlamaCpuContext`，terminal release 时释放请求的 Executor、Context 和 cache。

真实双请求 smoke 使用 3-token step budget，验证长 prompt 分块、22 层 K/V 隔离和终态回收：

```bash
cmake -S code/engine -B /tmp/cpp-llm-engine-real-serving \
  -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp" \
  -DCHAPTER06_SOURCE_DIR="$PWD/code/chapter06_quantization" \
  -DCPP_LLM_ENGINE_REAL_MODEL="$PWD/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-engine-real-serving \
  --target real_tinyllama_serving_smoke -j2
ctest --test-dir /tmp/cpp-llm-engine-real-serving \
  -R real_tinyllama_serving_test --output-on-failure
```

| 事件 | Service Core 动作 | Request 结果 |
| --- | --- | --- |
| `admit` | 校验 ID、terminal 状态和容量，保存 executor，进入 waiting | `Initial` |
| `dispatch_step` | 补充 running，按 token/block budget 执行一次 prefill chunk 或 decode | 每请求最多前进一步 |
| `cancel` / `timeout` | 移除 waiting 请求或把 running 请求标记为 terminal | `Cancelled` / `TimedOut` |
| `release_completed` | 清空 cache，释放 block、Executor 和 Context | terminal 且无请求资源 |
| 真实请求 A | `Hello TinyLlama` | `29892 306 29915 29885` |
| 真实请求 B | `Hi` | `839 744 29892 263` |
| 隔离与回收 | 检查 Context、cache、chunk 和指标 | `request_isolation_match`、`resources_released` 均为 true |

## 指标与时间线

`ServingCore::metrics()` 返回累计快照。请求结果使用 counter，延迟使用 Prometheus `_sum/_count`；平均值由 `sum / count` 计算。三种延迟使用 `steady_clock`，单元测试通过手动时钟固定时间，不依赖机器负载。

queue wait 从 admission 到进入 running；TTFT 到第一次 token callback；completion 到 finished、failed、cancelled 或 timed out。终态类型由对应 counter 区分。

| 指标 | 计数时机 |
| --- | --- |
| `accepted_total`、`queue_full_total`、`duplicate_total` | admission 结果 |
| `cancelled_total`、`timed_out_total`、`failed_total` | terminal 状态产生 |
| `completed_total`、`released_total` | 完成请求回收 |
| `generated_tokens_total`、`peak_inflight` | token 产生和并发峰值 |
| `queue_wait_seconds_sum/count` | admission 到进入 running |
| `ttft_seconds_sum/count` | admission 到首个 token callback |
| `completion_seconds_sum/count` | admission 到任一 terminal 状态 |

![Chapter 09 在线延迟时间线](../assets/diagrams/chapter09-latency-timeline.svg)

## HTTP 与后续扩展

`HttpAdapter` 把同步 completion、SSE token event 和错误状态映射到同一个 ServiceCore。Chapter05 callback 产生 token event；客户端断开映射为 `499/client_disconnect`，deadline 映射为 `408/timeout`，最终事件携带 `finish_reason`。

完成 CPU Service Core 后，扩展顺序是 GPU Cache Engine、并行 Batch Executor、Beast/Asio transport，再接只调用稳定 C++ 核心的 Python Gateway。下一页 [`09.2 Transport`](./02_transport.md) 继续验证网络库替换不会改变调度和资源所有权。

| 请求 | 成功响应 | 错误响应 |
| --- | --- | --- |
| `GET /health` | `200` JSON 健康状态 | - |
| `GET /metrics` | `200` Prometheus text | - |
| `POST /v1/completions` | `200` token ID、completion 和 finish reason | `400` body、`404` route、`429` capacity、`500` executor |
