# Repository Engine First Success 首个纵向切片

`code/engine` 把本教程自己的 CPU-only 纵向路径串成一个可运行切片，不替代 Chapter 00 的 llama.cpp 真实模型 baseline，也不是完整推理服务。

## 资产总览

| 资产 | 作用 |
| --- | --- |
| `include/` | toy Engine、真实 TinyLlama Engine、Tokenizer 和每请求执行器工厂接口 |
| `src/` | prompt、Tokenizer backend、教学 decoder 与真实 Model/Context 的运行时适配 |
| `tests/` | core contract、engine、embedding Runtime 与 ServiceCore 集成测试 |
| `EmbeddingRequestExecutor` | 预计算 text/visual embedding -> Chapter05 每请求执行器 |
| `examples/` | First Success、真实生成/服务/HTTP smoke 和真实服务 benchmark |
| `scripts/run.sh` | 默认隔离构建、demo 和 CTest 入口 |
| `scripts/run_real_serving_benchmark.sh` | 可选 TinyLlama 双请求 Release benchmark |
| `scripts/run_real_decoder_profile.sh` | 可选 TinyLlama Decoder 子阶段计时与结果归档 |
| `scripts/run_real_http_server.sh` | 启动真实 TinyLlama HTTP server，或以 `--check` 执行 curl 闭环 |
| `CMakeLists.txt` | CPU-only 集成构建 |

## 运行与验证

```bash
cmake -S code/engine -B code/engine/build
cmake --build code/engine/build
code/engine/build/cpp_llm_engine_demo --prompt "hello" --max-new-tokens 4
ctest --test-dir code/engine/build --output-on-failure
```

默认构建使用 toy config、toy tokenizer 和教学 decoder，适合快速回归接口与状态机。可选真实路径读取 TinyLlama GGUF，执行真实 tokenizer、22 层 CPU Decoder、greedy sampling 和 detokenize；真实服务 smoke 使用 3-token step budget 验证 append-prefill、双请求隔离和 Block KV 回收，但不代表生产并发性能。

`EmbeddingRequestExecutor` 接收 Core `DecoderEmbeddingSequence` 和每请求 `DecoderExecutionContext`，把预计算 embedding 适配为 Chapter05 `RequestModelExecutor`。Request 中的 prompt position 只承担序列长度和调度计数，真实 hidden values 由 Context 消费；prefill 成功后适配器立即释放 embedding storage，decode 继续使用同一 Context 和既有 KV 生命周期。默认测试用合成 Context 验证 Runtime 生成、长度校验、错误传播与释放，不要求多模态模型文件。

`embedding_request_service_test` 将同一个适配器注入 Chapter09 `HttpAdapter`。同步 completion 与真实 `ResponseWriter` SSE 都生成 `[2,4]`；压力段连续执行 32 个成功请求、8 个 prefill 失败请求，再覆盖客户端断开和 deadline 超时。每次终态后都检查 Context 析构、inflight 归零以及 accepted 与 completed/failed/cancelled/timed_out/released 计数。该测试固定服务复用契约，不解析图片 HTTP body，也不代替真实 SmolVLM 服务 benchmark。

Engine 已支持通过 `TokenizerBackend` 注入 tokenizer。默认路径使用 toy tokenizer；启用 llama.cpp 后，可验证真实 TinyLlama tokenizer 已进入 Engine 的 prompt、Request 和 Runtime 链路：

```bash
cmake -S code/engine -B /tmp/cpp-llm-engine-real-tokenizer \
  -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp"
cmake --build /tmp/cpp-llm-engine-real-tokenizer --target real_tokenizer_engine_smoke
/tmp/cpp-llm-engine-real-tokenizer/real_tokenizer_engine_smoke \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

这个 smoke 只验证 tokenizer contract。继续构建真实生成目标，可验证同一套 Chapter05 Runtime 驱动真实模型：

```bash
cmake -S code/engine -B /tmp/cpp-llm-engine-real-generation \
  -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp" \
  -DCHAPTER06_SOURCE_DIR="$PWD/code/chapter06_quantization" \
  -DCPP_LLM_ENGINE_REAL_MODEL="$PWD/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-engine-real-generation \
  --target real_tinyllama_generation_smoke real_tinyllama_serving_smoke \
  real_tinyllama_http_adapter_smoke -j2
/tmp/cpp-llm-engine-real-generation/real_tinyllama_generation_smoke \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  4 "Hello TinyLlama"
ctest --test-dir /tmp/cpp-llm-engine-real-generation \
  -R 'real_tinyllama_(generation|serving|http_adapter)_test' --output-on-failure
```

设置 `CPP_LLM_ENGINE_REAL_MODEL` 后才会注册标记为 `real_model;slow` 的真实模型 CTest，默认无模型构建仍只运行轻量测试。当前基线输出 6 个 prompt token，并生成 `29892 306 29915 29885`，解码文本为 `, I'm`。CTest 在同一 Engine 上连续生成两次，断言 `golden_tokens_match: true` 和 `context_isolation_match: true`。

`RealLlamaExecutorFactory` 长期持有只读 `RealLlamaCpuModel`，每次 `create_executor()` 创建一个独立 `RealLlamaCpuContext`。Factory 向外提供 Core `DecoderModelContract`，Executor 只持有 `DecoderExecutionContext`；Chapter05 Runtime 因而不依赖 GGUF loader 或真实 Decoder 的具体类型。Context 独占 position 和 22 层 K/V。真实服务 smoke 用同一 Factory 为 `Hello TinyLlama` 和 `Hi` 创建两个 Executor；长 prompt 分块追加到同一 Context，中间 chunk 不产生 logits，固定输出仍分别为 `29892 306 29915 29885` 与 `839 744 29892 263`。terminal release 后，两个 Executor/Context 和 block table 均被释放。当前仍是单线程 step 正确性验证，不宣称线程安全或并行吞吐。

`real_tinyllama_http_adapter_smoke` 把同一 Factory 注入 Chapter09 `HttpAdapter`，验证同步 JSON 返回第一组 token、SSE 按顺序返回第二组 token，并核对两个请求的 accepted/completed/released/generated metrics。需要继续验证真实 socket 时运行 `bash code/engine/scripts/run_real_http_server.sh --check`；去掉 `--check` 会保留服务供手工 curl。transport 只处理 HTTP/SSE，并用互斥锁串行调用当前单线程 ServiceCore。

真实服务性能基线使用独立脚本，默认执行一次 warmup 和三轮测量：

```bash
bash code/engine/scripts/run_real_serving_benchmark.sh
```

结果写入 `code/chapter09_serving/results/tinyllama_service_cpu_baseline.json`。比较前必须保持 CPU、编译器、Release 构建、模型、prompt 和 token budget 一致；正确性字段失败时，不使用延迟或吞吐数字做优化结论。

需要继续定位真实 CPU Decoder 内部耗时时，运行：

```bash
bash code/engine/scripts/run_real_decoder_profile.sh
```

该脚本只在专用 Context 上开启 profiling，默认生成路径不读取时钟。结果写入 `code/chapter08_profiling/results/tinyllama_decoder_phase_profile.txt`；只有 golden token、Decoder 顶层计时和 FFN 子项闭合全部通过，脚本才接受并归档结果。

profile CLI 的第三个参数可以选择 `scalar`、`q4-avx2`、`q6-avx2`、`q4-attention-avx2` 或 `optimized-avx2`。前三个实验模式分别只切换 FFN gate/up、FFN down 或 Attention QKV/out，组合模式同时启用三个已验证作用域并保持 LM Head scalar；正常生成和服务仍默认使用 scalar。Chapter08 用下面四个封装完成固定 workload A/B：

```bash
bash code/chapter08_profiling/scripts/run_real_q6k_down_ab.sh
bash code/chapter08_profiling/scripts/run_real_q4k_ffn_ab.sh
bash code/chapter08_profiling/scripts/run_real_q4k_attention_ab.sh
bash code/chapter08_profiling/scripts/run_real_optimized_policy_ab.sh
```

AVX2 路径会在选择时检查 AVX2 与 FMA，不能运行时直接返回错误，不静默回退。
