# 07.6 主流对照：把 benchmark 放回真实服务里看

到这里，你已经把正确性和性能拆开了。  
这一节不再重复讲测试和 benchmark，而是把本章得到的结果放到 `vLLM`、`SGLang`、`TensorRT-LLM` 和 `Ollama` 里对照着看，帮你分清：

- 哪些结果是在比 runtime
- 哪些结果是在比服务方式
- 哪些结果是在比 engine 和交付方式

## 1. 本节概览

| 本章里的东西 | `vLLM` 里会看什么 | `SGLang` 里会看什么 | `TensorRT-LLM` 里会看什么 | `Ollama` 里会看什么 |
| --- | --- | --- | --- | --- |
| 正确性测试 | 高吞吐 serving 下链路是否还对 | 多步编排下结果是否稳定 | engine 接入后结果是否一致 | 本地交付后是否还能稳定使用 |
| benchmark 指标 | 吞吐、延迟、cache 行为 | 应用流程下的响应特征 | GPU engine 的性能边界 | 本地打包后的实际体验 |
| baseline / optimized / reference | serving runtime 的对照 | 编排方式的对照 | engine / runtime 路径的对照 | 本地交付方式的对照 |
| 结果解读 | 生成和调度是否真的变快 | 多步骤流程是否仍然可控 | 优化是否真的来自 engine | 打包后是否仍然好用 |

## 2. 为什么它重要

本章已经把结果拆成了三层：

- 单元测试看局部正确性
- 端到端测试看完整链路
- benchmark 看固定条件下的性能比较

但光把数字跑出来还不够。  
你还要知道这些数字该放到哪类系统里看，才知道它们到底说明了什么。

`vLLM`、`SGLang`、`TensorRT-LLM` 和 `Ollama` 分别对应四种常见场景：

1. 高吞吐 serving
2. 应用编排
3. 高性能 engine
4. 本地交付

这也是这节要做的事：把 chapter07 的 benchmark 结果放回真实使用场景里看，而不是只在教程内部看一组数字。

## 3. 先看什么

如果你先想建立一个最小判断顺序，可以这么看：

1. 先看 `vLLM`，它最接近本章的 serving 结果比较
2. 再看 `SGLang`，它更像把生成能力组织成流程以后会发生什么
3. 再看 `TensorRT-LLM`，它最适合用来理解 engine / runtime 的性能边界
4. 最后看 `Ollama`，它更像把结果交给用户之后的最终体验

这四个方向里，`vLLM` 最像 chapter05 + chapter07 的组合结果，`TensorRT-LLM` 最像 chapter03 + chapter04 + chapter06 的组合结果，`SGLang` 和 `Ollama` 则提醒你，benchmark 之外还有编排和交付。

## 4. 中间推导是什么

往回推为什么要把这些框架放进来，通常会得到几个约束：

- benchmark 不是单独成立的，它必须跟实际使用场景一起看
- 吞吐和延迟不是抽象数字，它们会受 serving 方式影响
- engine、runtime、编排和交付，都会影响最后的结果解读
- 如果不把结果放回生态里，数字很容易只剩下比较，没有结论

把这些约束摆出来后，就能看懂为什么 chapter07 要做的不只是跑分，而是把结果变成证据：

- `vLLM` 帮你看高吞吐 serving
- `SGLang` 帮你看应用编排
- `TensorRT-LLM` 帮你看 engine 级优化
- `Ollama` 帮你看本地交付后的实际使用感受

## 5. 你现在应该能做什么

学完本节后，你应该已经能把下面几件事说清楚：

- 为什么同一组 benchmark 数字，在不同场景下含义不一样
- 为什么 `vLLM` 更适合看吞吐和调度，`TensorRT-LLM` 更适合看 engine 边界
- 为什么 `SGLang` 更适合看多步流程，而 `Ollama` 更适合看交付体验
- 为什么 chapter07 的结果不能只在教程内部自洽，还要放回真实生态里判断

把这些对应关系讲明白，就说明你已经能把 benchmark 结果放回真实使用场景里去看。

## 6. 边界说明

这一节不负责复刻 `vLLM`、`SGLang`、`TensorRT-LLM` 或 `Ollama` 的全部能力。

它只负责把本章的验证结果和这些真实场景的关注点对齐，让你知道：

- 哪些数字是在比 runtime
- 哪些数字是在比编排
- 哪些数字是在比 engine
- 哪些数字是在比交付

## 7. 参考页

- [vLLM](/reference/ecosystem/vllm)
- [SGLang](/reference/ecosystem/sglang)
- [TensorRT-LLM](/reference/ecosystem/tensorrt_llm)
- [Ollama](/reference/ecosystem/ollama)
- [chapter07_benchmark 代码入口](/code/chapter07_benchmark)
