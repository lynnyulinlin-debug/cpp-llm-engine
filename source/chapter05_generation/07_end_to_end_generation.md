# 05.7 End-to-End Generation Testing 端到端生成测试

## 验证完整生成链路

完成一次端到端生成测试：输入 prompt，输出固定数量的 token，确认生成链路完整可用。

**阅读入口**

- 前置阅读：[`05.4 KV Cache`](./04_kv_cache.md)、[`05.5 Scheduler`](./05_scheduler.md)、[`05.6 Sampler`](./06_sampler.md)
- 源码直达：<a href="../_code/chapter05_generation/src/runtime.cpp"><code>runtime.cpp</code></a>、<a href="../_code/chapter05_generation/tests/chapter05_generation_test.cpp"><code>chapter05_generation_test.cpp</code></a>、<a href="../_code/chapter05_generation/tests/runtime_smoke_test.cpp"><code>runtime_smoke_test.cpp</code></a>
- 相关阅读：[`章节导读`](./intro.md)、[`chapter06_quantization`](../chapter06_quantization/intro.md)

## 测试设计

前面的模块分别正确，并不代表组合起来也正确。端到端测试就是最终验收点。

端到端测试要确认：输入能进、Request 状态能走、token history 能检查、真实 Context 能隔离分层 K/V、sampler 能出 token，最后还能稳定回到一个可检查的输出。

最小测试只需要固定两件事：prompt 和生成 token 数量。这样就能把它当成一条稳定的回归基线，而不是一次性的手工演示。

端到端测试的成功标准：

| 标准 | 说明 |
| --- | --- |
| 无崩溃 | 生成过程中没有内存错误或异常 |
| 输出长度正确 | 生成了 `max_new_tokens` 个 token |
| 采样可复现 | 在 greedy 模式下，多次运行结果一致 |
| 非空输出 | 至少生成了一个 token |
| 停止条件触发 | 达到 `max_new_tokens` 或生成 EOS |

最低通过标准是：输出非空且长度正确。进阶标准是：在 greedy 模式下，输出与记录过的黄金输出完全一致。

黄金输出是什么：

黄金输出（golden output）是用于回归测试的稳定参考。下面使用 GoogleTest 风格断言表达测试意图，不是仓库现有测试框架的可直接编译代码；仓库测试使用标准 `assert` 和 CTest：

```cpp
// 第一次运行：记录输出
std::string prompt = "Hello";
auto output = generate(prompt, 16);
std::string expected = output;

// 后续运行：对比输出
auto new_output = generate(prompt, 16);
EXPECT_EQ(new_output, expected);
```

注意：

- 黄金输出最适合 greedy 模式，因为它是确定性的
- 如果使用 temperature 采样，通常要固定 seed 才能复现
- 模型更新后，黄金输出可能需要重新生成

端到端测试放在缓存对象说明之后，因为它需要同时验证教学 Runtime 的 token history 和真实 Executor Context 的分层 K/V。前者证明 Request 状态可检查，后者证明 prefill/decode 能增量推进且请求互不污染。

如果这里失败，通常需要回头检查 embedding、decoder、cache 或 sampler 其中某一段。

最小测试流程也用概念断言表示：

```cpp
std::string prompt = "Hello";
auto output = generate(prompt, /*max_new_tokens=*/16);
EXPECT_FALSE(output.empty());
EXPECT_EQ(output.size(), 16);
```

这里的重点是“链路闭环”，不是生成质量。

Sampler 的集成验证：

Sampler 的验证通常可以通过 greedy 模式来做。下面的 `Config` 和 `generate` 是接口形状伪代码；实际接口请以 `GenerationConfig`、`Request`、`GreedySampler` 和 `Runtime::generate()` 为准：

```cpp
auto config = Config();
config.sampler.temperature = 0.0f;  // greedy
config.sampler.top_k = 0;           // 禁用 top-k

auto output1 = generate("Hello", config, 16);
auto output2 = generate("Hello", config, 16);
EXPECT_EQ(output1, output2);
```

如果两次输出不一致，说明 sampler 里还有非确定性因素，比如随机种子没有固定。

## 真实 TinyLlama 生成验证

教学测试稳定后，再用同一个 `Runtime::generate` 接口替换 logits 来源。`RealTokenizerBackend` 负责文本与 token 的转换，`RealLlamaExecutorFactory` 持有共享的 22 层 `RealLlamaCpuModel`，每次生成创建的 `RealLlamaCpuContext` 负责 prefill/decode 和分层 K/V；`Runtime` 负责 Request 状态、greedy sampling、EOS 和 token budget。

从仓库根目录执行：

```bash
cmake -S code/engine -B /tmp/cpp-llm-engine-real-generation \
  -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp" \
  -DCHAPTER06_SOURCE_DIR="$PWD/code/chapter06_quantization" \
  -DCPP_LLM_ENGINE_REAL_MODEL="$PWD/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-engine-real-generation \
  --target real_tinyllama_generation_smoke -j2
/tmp/cpp-llm-engine-real-generation/real_tinyllama_generation_smoke \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  4 "Hello TinyLlama"
ctest --test-dir /tmp/cpp-llm-engine-real-generation \
  -R real_tinyllama_generation_test --output-on-failure
```

当前 smoke 是单请求 CPU 正确性验证，不测并发吞吐，也不经过 Chapter09 HTTP transport。固定模型文件与 prompt 后，greedy 输出可作为后续重构的回归记录；同一 Factory 接入 ServiceCore 的双请求验证见 [`Chapter09 Service Core`](../chapter09_serving/01_service_core.md)。

| 检查项 | 固定结果 |
| --- | --- |
| prompt | `Hello TinyLlama` |
| prompt token 数 | `6` |
| 生成 token | `29892 306 29915 29885` |
| detokenize 文本 | `, I'm` |
| golden CTest | `golden_tokens_match: true` |
| 连续请求隔离 | `context_isolation_match: true` |
| 停止条件 | 达到 `max_new_tokens=4`；若先生成 EOS 则提前停止 |

## 验证与衔接

验证时重点看：

- 程序可以稳定生成 token
- 输出数量达到预期
- 没有在中途崩溃或卡死
- 同一 prompt 下，结果能稳定复现
- greedy 模式下可以得到黄金输出或完全一致的回归结果

一个好的端到端测试应当能在固定输入下稳定复现同类行为，这样后续重构才有可比基线。

失败时怎么排查：

端到端测试失败时，可以按这个顺序往回查：

1. 检查输出长度
   - 长度 = 0，说明生成循环没真正跑起来
   - 长度 < 预期，说明可能提前遇到 EOS
   - 长度 > 预期，说明停止条件没触发
2. 检查 cache
   - 单独对比 naive 和 cache 版本的输出
3. 检查 sampler
   - 确认 logits 的 shape 是 `[vocab_size]`
   - 确认 greedy 模式下是确定性的
4. 检查 embedding 和 decoder 的 shape 一致性
   - embedding 输出应是 `[B, S, D]`
   - decoder 输入也应是 `[B, S, D]`

常见问题：

- 生成数量配置错误
- prompt 编码错误
- 前面某个模块的 shape 问题在这里被放大
- 只验证“有输出”，没有验证“输出长度和停止条件是否受控”
- 把端到端测试当成质量评估，而不是链路闭环检查
- 一旦结果异常，无法判断问题是出在 cache 还是 sampler

和前面的关系：

端到端测试把前面的模块串成一个可回归闭环：

- `01_embedding_and_position` 负责输入如何进入模型
- `04_kv_cache` 负责历史状态如何保存和复用
- 端到端测试负责验证链路是否真的闭合

如果端到端测试不稳定，说明 generation runtime 还不能算稳定闭合。
