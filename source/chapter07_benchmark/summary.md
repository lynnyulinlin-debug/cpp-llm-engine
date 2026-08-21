# 07.8 Chapter Wrap-up 本章解决了什么

这章回答的是：正确性和性能是否已经可以被系统地验证、对照和记录。到这里，测试、比较和结果解读已经收成一条可复查的证据链。

## 本章解决了什么

- 正确性和性能已经被拆成两条线
- 单元测试和端到端测试负责证明链路没有被做坏
- benchmark 负责固定条件下的性能比较
- 回归样例负责固定 prompt、边界输入和历史行为
- 结果可以按 baseline、optimized 和 reference 三层来拆
- benchmark 数字已经能被翻译成保留、回滚或继续优化的决策

## 为什么重要

这一章把整套教程的收口方式固定下来：前面章节负责把系统做出来，这一章负责把它变成可验证、可比较、可决策的证据。

## 下一步

进入 `chapter08_profiling`，做 profiling 和 A/B 对比。`chapter07` 留下的是可比较结果，`chapter08` 要解释这些结果背后的原因。

## 复盘路径

- 先看 [`07.6 Result Interpretation`](./06_result_interpretation.md)
- 再看 [`07.7 主流对照`](./07_framework_comparison.md)
- 对照 [`vLLM`](../reference/ecosystem/vllm.md) 回看高吞吐 serving
- 对照 [`SGLang`](../reference/ecosystem/sglang.md) 回看应用编排
- 对照 [`TensorRT-LLM`](../reference/ecosystem/tensorrt_llm.md) 回看 engine 级优化
- 对照 [`Ollama`](../reference/ecosystem/ollama.md) 回看本地交付体验

这一章不负责解释所有性能波动。它只负责把验证体系收口成一套稳定流程：正确性由测试承担，历史行为由回归样例承担，性能由 benchmark 承担，结果解读由 `07.6` 的决策模板承担。

profiling 不等于 benchmark。benchmark 负责告诉你“发生了什么”，profiling 负责告诉你“为什么会这样”。

- 先看 <a href="../_code/chapter07_benchmark/src/benchmark.cpp"><code>benchmark.cpp</code></a>
- 再看 <a href="../_code/chapter07_benchmark/src/test_helpers.cpp"><code>test_helpers.cpp</code></a>
- 最后看 <a href="../_code/chapter07_benchmark/tests/chapter07_benchmark_test.cpp"><code>chapter07_benchmark_test.cpp</code></a>
