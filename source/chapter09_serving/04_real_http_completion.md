# 09.4 真实模型 HTTP Completion

`HttpAdapter` 通过可选的 `RequestExecutorFactory` 接收真实模型执行器。应用启动时加载一次 `RealLlamaExecutorFactory`；每个 completion 到达后，adapter 调用一次 Factory，创建独立的 Executor/Context，再把它连同 Request 交给 `ServiceCore`。Chapter09 因此仍只依赖通用执行接口，不依赖 TinyLlama、GGUF 或 llama.cpp。

当前最小 HTTP body 接收 `prompt_tokens`。真实 tokenizer 位于 Engine 应用组合层：它先把文本转成 token ID，再构造 HTTP 请求。后续增加文本 API 时，可以在 Gateway 增加 `prompt -> tokenizer -> prompt_tokens`，无需修改 ServiceCore、Scheduler 或模型 Context 的所有权。

## 请求路径

同步和 SSE 使用同一个 Factory 与服务核心，区别只在输出方式。同步响应返回 `generated_token_ids`，便于直接核对 golden token；SSE 通过 `ResponseWriter` 按生成顺序写出 token event。Factory 返回空指针时，请求在 admission 前返回 `500/model_executor_unavailable`。

| 阶段 | 对象 | 可验证结果 |
| --- | --- | --- |
| 应用启动 | `RealTokenizerBackend`、`RealLlamaExecutorFactory` | tokenizer 与模型词表大小一致，Model 只加载一次 |
| 请求创建 | `RequestExecutorFactory` | 每个请求创建独立 Executor/Context |
| 同步 completion | `HttpAdapter::handle()` | 返回固定 `generated_token_ids` 和 `finish_reason` |
| 流式 completion | `HttpAdapter::handle_stream()` | token event 顺序固定，最后写出 `stop` |
| 终态回收 | `ServiceCore::release_completed()` | Request、KV Cache 和 Executor/Context 释放 |
| 指标读取 | `GET /metrics` | accepted、completed、released 和 token 总数一致 |

## 真实模型验证

下面的 target 不启动 socket，专门验证“真实模型 + HTTP 语义 + ServiceCore”的应用组合。cpp-httplib transport 继续复用同一个 `HttpAdapter`，不会获得另一套模型执行逻辑。

```bash
cmake -S code/engine -B /tmp/cpp-llm-engine-real-http \
  -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp" \
  -DCHAPTER06_SOURCE_DIR="$PWD/code/chapter06_quantization" \
  -DCPP_LLM_ENGINE_REAL_MODEL="$PWD/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-engine-real-http \
  --target real_tinyllama_http_adapter_smoke -j2
ctest --test-dir /tmp/cpp-llm-engine-real-http \
  -R real_tinyllama_http_adapter_test --output-on-failure
```

| 请求 | 输出路径 | 固定结果 |
| --- | --- | --- |
| `Hello TinyLlama` | 同步 JSON | `29892 306 29915 29885` |
| `Hi` | SSE event | `839 744 29892 263` |
| 两个请求累计 | `/metrics` | accepted/completed/released=`2`，generated tokens=`8`，三组延迟 count=`2` |

## 本地 HTTP 闭环

真实监听入口复用上述 Model、Factory、ServiceCore 和 HttpAdapter。请求仍接收 `prompt_tokens`；cpp-httplib 只负责 socket 与 SSE 写出，transport 用互斥锁串行调用当前单线程 ServiceCore。

从仓库根目录执行一条命令，可以构建 server、临时启动、运行四组 curl 检查并退出：

```bash
CPP_LLM_ENGINE_MODEL_PATH=/path/to/tinyllama.gguf \
bash code/engine/scripts/run_real_http_server.sh --check
```

需要保留服务供手工调用时，去掉 `--check`。默认监听 `127.0.0.1:18082`，可通过 `REAL_HTTP_BIND_ADDRESS` 和 `REAL_HTTP_PORT` 修改。最小同步请求如下：

```bash
curl --fail --silent \
  -H 'Content-Type: application/json' \
  -d '{"request_id":101,"prompt_tokens":[15043,18439,1460,29931,5288,29874]}' \
  http://127.0.0.1:18082/v1/completions
```

`--check` 依次验证 `/health`、固定 golden token 的同步 completion、相同 token 顺序的 SSE completion，以及 accepted/completed/released/generated metrics。文本 `prompt` 到 token ID 的转换仍属于后续 Gateway，不在 transport 中复制 tokenizer 逻辑。
