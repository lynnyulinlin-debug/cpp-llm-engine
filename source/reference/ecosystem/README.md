# 推理生态项目索引

这里是框架对照的项目卡片层。先在[训练框架、推理引擎与推理服务](../framework_comparison.md)确定当前问题所在的系统位置，再选择一个项目下钻；项目卡不组成第二条学习主线。

## 按问题选择项目

同一个项目可能跨越多层，但每张卡只保留它对本教程最有价值的阅读角度。模型家族和交付工具单独列出，避免与推理引擎混为一类。

| 当前问题 | 优先阅读 | 补充阅读 |
| --- | --- | --- |
| 本地真实模型与 GGUF | [llama.cpp](./llama_cpp.md) | [Gemma](./gemma.md) |
| C++/CUDA 纵向实现 | [Tiny-VLLM](./tiny_vllm.md) | [TensorRT-LLM](./tensorrt_llm.md) |
| 可读的模型与 Runtime 算法 | [nano-vLLM](./nano_vllm.md) | [vLLM](./vllm.md) |
| Continuous Batching 与 Block KV | [vLLM](./vllm.md) | [Mini-SGLang](./mini_sglang.md) |
| Radix Cache、LPM 与前缀复用 | [Mini-SGLang](./mini_sglang.md) | [SGLang](./sglang.md) |
| GPU kernel、量化和 engine build | [TensorRT-LLM](./tensorrt_llm.md) | [Tiny-VLLM](./tiny_vllm.md) |
| 大规模阶段分离与联合优化 | [DeepInfer](./deepinfer.md) | [SGLang](./sglang.md) |
| 本地产品交付 | [Ollama](./ollama.md) | [llama.cpp](./llama_cpp.md) |

## 项目类型

分类只用于避免误读，不代表项目只能承担一种职责。

| 类型 | 项目 | 阅读目的 |
| --- | --- | --- |
| 本地 C++ baseline | llama.cpp | 建立真实推理、GGUF 和 CPU 路径坐标 |
| 教学型实现 | Tiny-VLLM、nano-vLLM、Mini-SGLang | 观察被压缩后的 kernel、模型或 Runtime 骨架 |
| 生产推理框架 | vLLM、SGLang、TensorRT-LLM、DeepInfer | 观察规模扩大后的执行、调度、缓存和部署约束 |
| 模型家族 | Gemma | 观察模型结构怎样影响 loader、shape 和量化 |
| 交付工具 | Ollama | 观察推理能力怎样包装为本地产品 |

## 卡片写作规则

项目卡固定回答三个问题：为什么现在读、重点看哪些对象、与教程哪些章节对应。项目版本快速变化时只维护官方入口和稳定概念，不在卡片中复制大段源码或短期 benchmark 数字。
