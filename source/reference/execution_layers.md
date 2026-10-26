# Model、算子、Backend、Context 与 Runtime

执行路径从只读 Model 开始，经过 operator/backend 完成数值计算，由请求 Context 保存状态，再由 executor/runtime 推进生成。

## 1. 这页要解决什么

目标是确定模型结构、数值计算、请求状态和调度分别放在哪里。

如果四层边界不清楚，后面最容易出现的问题是：

- Model 和 Context 共享可变请求状态
- Runtime 直接依赖 GGUF loader 或具体 Decoder 类型
- backend 改变 operator 语义
- 调度逻辑进入算子和 kernel

完整系统中的位置见 [推理引擎系统补强图](./inference_engine_systems.md)。执行层参考展开其中的计算与执行路径。

## 2. 六个执行对象

### model

`model` 持有模型配置和只读权重，并创建请求 Context。`DecoderModelContract` 让 Engine 在不读取 GGUF 的情况下检查词表、层数、上下文长度和单层结构。

### operator

`operator` 负责的是**语义**。

它要回答的是：

- 这个计算做什么
- 输入输出是什么
- shape 和布局约束是什么
- 错误边界在哪里

在教程里，`MatMul` 就是最典型的 `operator`。

### backend

`backend` 负责的是**具体实现路径**。

它要回答的是：

- 这件事用 CPU 还是 GPU 做
- 用参考实现还是库实现
- 用 cuBLAS 还是未来其他 kernel

backend 不应该改写上层语义，只应该实现同一套语义。

### executor

`executor` 负责的是**把某次执行真正跑起来**。

它要回答的是：

- 这次执行调用哪个 backend
- 输入怎么喂进去
- 输出怎么拿出来
- 这次执行的资源边界是什么

教程中的 Engine adapter 把 Core `DecoderExecutionContext` 转为 Chapter05 `RequestModelExecutor`，不复制 Decoder 数学。

### context

`context` 保存一次请求的 position、分层 K/V 和临时统计。`DecoderExecutionContext` 提供 `reset`、完整 `prefill_tokens`、增量 `prefill_chunk`、`decode_token` 和 `token_count`；中间 chunk 只追加状态，final chunk 才返回 logits。每个运行中请求独占一个 Context。

### runtime

`runtime` 负责的是**持续服务和状态推进**。

它要回答的是：

- 请求怎么进入系统
- `prefill` 和 `decode` 怎么推进
- `KV Cache` 怎么复用
- scheduler 怎么决策
- 多请求怎么持续工作

runtime 关心的是系统，不只是单次执行。

## 3. 一条最小分工链

| 对象 | 主要职责 | 典型问题 |
| --- | --- | --- |
| `model` | 持有只读权重和结构 | 这个模型能创建什么 Context |
| `operator` | 定义语义边界 | 这一步算的是什么 |
| `backend` | 实现具体计算 | 这一步怎么高效算 |
| `context` | 保存请求级执行状态 | 当前请求执行到哪个 token |
| `executor` | 适配 Runtime 调用 | Runtime 如何取得下一步 logits |
| `runtime` | 持续推进请求 | 系统怎么一直服务下去 |

## 4. 和主线章节的关系

| 章节 | 已经做了什么 | 主要执行对象 |
| --- | --- | --- |
| `chapter01_tensor_loader` | GGUF、真实 Model/Context、TinyLlama 对照 | `model` + `context` 实现 |
| `chapter03_matmul` | 算子边界、CPU reference、cuBLAS | `operator` + `backend` |
| `chapter04_attention` | Attention、FFN、单层 Decoder、结构契约 | `operator` + 模型结构 |
| `chapter05_generation` | 生成链路、request、scheduler、cache | `runtime` |
| `code/engine` | Core Context 到 Runtime executor 的适配 | `executor` |
| `chapter07_benchmark` | 正确性和性能收口 | `executor` + `runtime` 的验证结果 |

## 5. 如何使用执行层参考页

- 学完 Chapter04 后，用 `DecoderLayerContract` 对照教学层和真实 TinyLlama 的结构差异。
- 学完 Chapter05 后，沿 `Model -> Context -> Engine adapter -> Runtime` 跟踪一次请求。
- 扩展 CPU/GPU kernel 时只替换 backend；扩展调度时保持 Context 执行接口稳定。
