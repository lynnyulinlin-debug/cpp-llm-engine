# 05.9 Request 生命周期

生成 runtime 不是一次函数调用，而是一批请求在多个 step 之间不断流转。`Request` 在这里从“输入参数”提升为运行时对象，用于说明 vLLM / nano-vLLM 里的 scheduler 到底在调度什么。

![Request 状态与资源生命周期](../assets/diagrams/chapter05-runtime-state.svg)

**阅读入口**

- 前置阅读：[`05.3 Naive Generation`](./03_naive_generation.md)、[`05.5 Scheduler`](./05_scheduler.md)
- 源码直达：<a href="../_code/chapter05_generation/src/request.cpp"><code>request.cpp</code></a>、<a href="../_code/chapter05_generation/src/scheduler.cpp"><code>scheduler.cpp</code></a>
- 相关阅读：[`05.10 Continuous Batching 入门`](./09_continuous_batching.md)、[`推理引擎输入链路`](../reference/input_pipeline.md)、[`推理引擎服务化基础`](../reference/service_basics.md)

## 一个请求最少要记录什么

| 字段 | 为什么需要 |
| --- | --- |
| prompt tokens | prefill 的输入来源 |
| prefill offset | 已成功写入模型 Context 的 prompt token 数 |
| generated tokens | decode 已经产出的状态 |
| max tokens / stop tokens | 判断何时结束 |
| sampling params | 控制 logits 到 token 的选择方式 |
| token history | 记录 Runtime 已提交的 prompt/generated token |
| Model Context | 保存 position 和真实分层 K/V，由 executor 按请求隔离 |
| BlockTable | 记录逻辑 token 的物理 block ownership，不保存 K/V 数值 |
| status / finish reason | 让 scheduler 知道请求还能不能继续推进 |

在最小实现里，这些字段可以很朴素；在真实 serving 项目里，它们会继续扩展成 sequence、sequence group、request output、streaming 状态等结构。服务层只负责把外部请求转换成这些字段，不直接管理模型内部的计算对象。

## 生命周期怎么流转

代码中的生命周期状态如下：

1. `Initial`：请求已创建；chunked prefill 期间仍保持该状态，由 `prefill_offset` 表示部分进度。
2. `Prefilled`：prompt tokens 已送入模型，Request history 和 Model Context 已提交。
3. `Decoding`：每个 decode step 推进一个新 token。
4. `Finished`：达到生成上限，可以返回结果。
5. `Cancelled` / `TimedOut`：请求被终止，scheduler 不再推进。
6. `Rejected`：admission budget 不满足，请求没有执行，也不应占有 Context 或 block。
7. `Failed`：模型 executor 的 prefill/decode 失败或返回空 logits，不再参与调度。

这个生命周期解释了为什么 scheduler 不能只保存一个待生成列表。它需要知道每个请求处于哪一步、占了多少 cache、下一步是否还能继续执行。

服务化场景还要补两条规则：

- `Cancelled` / `TimedOut` 请求不能再次进入 prefill 或 decode。
- `Rejected` 请求不能进入 prefill 或 decode，也不应拥有已提交的 token history、Context K/V 或 block table。
- `Failed` 请求必须保存可检查的 failure reason，不能用占位 token 掩盖执行错误。
- 进入任意 terminal 状态后，Request cache、请求 Context、workspace 和流式输出状态都必须进入回收路径。

## 对照 vLLM / nano-vLLM 时看什么

先找请求对象，再找它如何进入等待队列和运行队列。不要一开始就追所有优化策略，先确认三个问题：

- 一个请求什么时候从 waiting 进入 prefill？
- prefill 结束后，token history、模型 K/V 和 block ownership 分别保存在哪里？
- decode 每一步生成 token 后，请求状态在哪里更新？

这三个问题构成 serving runtime 的主线。后面的连续批处理、抢占、block 管理和 streaming 输出，都是在这个生命周期上继续增加规则。

相关阅读：[`推理引擎服务化基础`](../reference/service_basics.md)、[`推理引擎内存管理`](../reference/memory_management.md)
