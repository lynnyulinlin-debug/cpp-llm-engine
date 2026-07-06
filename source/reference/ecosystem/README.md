# 生态对照

这个目录只做一件事：把常见推理 / serving 方案放到同一张图里，方便读者快速判断该看谁、为什么看、和教程哪一章对照。

如果你想先理解教程主线，请优先看 [docs/README.md](/README) 和 [阅读指南](/guide)。  
这里是查阅层，不是第二条阅读主线。

## 一张表看懂借鉴关系

| 对象 | 类型 | 最强项 | 本教程最值得借鉴的点 | 对应章节 |
| --- | --- | --- | --- | --- |
| `llama.cpp` | 本地推理 baseline | 小、轻、可移植，适合先把模型跑起来 | 先做最小可用链路，先验证环境、加载和首次推理，再谈扩展 | `chapter00_quickstart`、`chapter01_tensor_loader`、`chapter05_generation` |
| `gemma.cpp` | 具体模型的 C++ 示例 | 作用域更窄，路径更短，适合初学者跟读 | 用一个具体模型家族看懂“模型如何映射到 C++ 推理流程” | `chapter01_tensor_loader`、`chapter02_tensor_ops`、`chapter04_attention` |
| `vLLM` | 高吞吐 serving runtime | continuous batching、scheduler、KV cache 管理、streaming | 学它的调度思想、缓存组织和吞吐/延迟平衡 | `chapter05_generation`、`chapter07_benchmark` |
| `SGLang` | 应用编排 / structured serving | 多步流程、结构化生成、上层 workflow 组织 | 学它的“把推理能力组织成应用运行时”的思路 | `chapter05_generation`、`chapter07_benchmark` |
| `TensorRT-LLM` | GPU engine / deployment reference | engine build、runtime 分层、GPU kernel 和量化部署 | 学它如何把“算子层 -> runtime -> 部署”串成一条高性能工程线 | `chapter03_matmul`、`chapter04_attention`、`chapter06_quantization`、`chapter07_benchmark` |
| `Gemma` | 模型家族 | 具体模型结构、兼容性和验证对象 | 学它作为“被推理的对象”如何影响 loader、shape、quantization 和 benchmark | `chapter01_tensor_loader`、`chapter04_attention`、`chapter06_quantization`、`chapter07_benchmark` |
| `Ollama` | 本地打包 / 分发层 | 交付简单、上手快、分发友好 | 学它如何把推理能力包装成更容易使用的本地产品 | `chapter00_quickstart`、`chapter05_generation` |

## 需要区分的两种对象

为了避免混淆，这里把两类对象分开：

- **模型家族**：`Gemma`
  - 它是“被推理的对象”
  - 用来验证 loader、shape、quantization 和 benchmark
- **推理 / serving 参考**：`llama.cpp`、`gemma.cpp`、`vLLM`、`SGLang`、`TensorRT-LLM`
  - 它们是“怎么推理、怎么服务、怎么组织运行时”的参考

## 学习顺序建议

如果你沿着本教程主线学习，建议按这个顺序看：

1. `llama.cpp`
   - 先建立最小本地推理直觉
2. `gemma.cpp`
   - 再看一个更具体、路径更短的 C++ 实现案例
3. `vLLM`
   - 然后看高吞吐 runtime 如何做调度和缓存
4. `TensorRT-LLM`
   - 再看高性能 GPU engine 如何分层
5. `SGLang`
   - 最后看上层应用编排

这个顺序和教程主线是对齐的：

- `chapter00_quickstart` 最适合先看 `llama.cpp`
- `chapter01_tensor_loader` 到 `chapter04_attention` 很适合看 `gemma.cpp` 和 `TensorRT-LLM`
- `chapter05_generation` 和 `chapter07_benchmark` 最适合看 `vLLM` 和 `SGLang`

## 参考文件

- `llama_cpp.md`
- `gemma.md`
- `tensorrt_llm.md`
- `vllm.md`
- `sglang.md`
- `ollama.md`
- `gemma.md`

## 写作规则

每一页都尽量短，直接回答三个问题：

- 它解决什么问题
- 它最值得借鉴的工程选择是什么
- 它和教程哪一章对应

每一页保持参考页风格，不展开成主线正文。
