# Further Reading 进一步阅读

这个页面收集主教程之外最值得继续看的内容。

## 1. 如果你想加深主线

- 回看 `chapter02_tensor_ops`，重点关注 view 语义和存储共享
- 回看 `chapter03_matmul`，重点关注算子边界和后端拆分
- 回看 `chapter04_attention`，重点关注 decoder 结构
- 回看 `chapter05_generation`，重点关注 runtime state、KV Cache 和 sampling

## 2. 如果你想走向 vLLM-like serving

- 阅读 `reference/ecosystem/vllm.md`
- 对照 request scheduling、batching、streaming 和 cache management
- 思考哪些部分可以迁移到你自己的 runtime

## 3. 如果你想继续看性能

- 阅读 `reference/ecosystem/tensorrt_llm.md`
- 回看 `chapter07_benchmark`，重点关注测量和比较纪律
- 在优化前先分清正确性、性能和 profiling

## 4. 如果你想扩展模型或交付视角

- 阅读 `reference/ecosystem/llama_cpp.md`
- 阅读 `reference/ecosystem/ollama.md`
- 阅读 `reference/ecosystem/gemma.md`

## 5. 如果你想理解训练框架和推理框架的分工

- 阅读 `reference/framework_comparison.md`
- 先分清训练、推理和本教程各自负责哪一层
- 再回到主线章节看对象、算子和 runtime 是怎么被拆开的

## 6. 如果你想补背景

- `guides/cmake_guide.md`
- `guides/setup_cuda.md`
- `guides/gguf_parser.md`
- `reference/faq.md`

## 7. 如果你想先统一术语

- `reference/glossary.md`
- 先把 `Tensor`、`MatMul`、`KV Cache`、`scheduler` 这类核心词的写法统一
