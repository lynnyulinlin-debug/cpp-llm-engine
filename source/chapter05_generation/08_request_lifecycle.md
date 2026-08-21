# 05.9 Request 生命周期

生成 runtime 不是一次函数调用，而是一批请求在多个 step 之间不断流转。本页把 `Request` 从“输入参数”提升为运行时对象，用于说明 vLLM / nano-vLLM 里的 scheduler 到底在调度什么。

## 阅读入口

- 前置阅读：[`05.3 Naive Generation`](./03_naive_generation.md)、[`05.5 Scheduler`](./05_scheduler.md)
- 源码直达：<a href="../_code/chapter05_generation/src/request.cpp"><code>request.cpp</code></a>、<a href="../_code/chapter05_generation/src/scheduler.cpp"><code>scheduler.cpp</code></a>
- 相关阅读：[`05.10 Continuous Batching 入门`](./09_continuous_batching.md)、[`推理引擎输入链路`](../reference/input_pipeline.md)、[`推理引擎服务化基础`](../reference/service_basics.md)

## 一个请求最少要记录什么

| 字段 | 为什么需要 |
| --- | --- |
| prompt tokens | prefill 的输入来源 |
| generated tokens | decode 已经产出的状态 |
| max tokens / stop tokens | 判断何时结束 |
| sampling params | 控制 logits 到 token 的选择方式 |
| KV cache view | 指向这个请求已经写入的历史状态 |
| status / finish reason | 让 scheduler 知道请求还能不能继续推进 |

在最小实现里，这些字段可以很朴素；在真实 serving 项目里，它们会继续扩展成 sequence、sequence group、request output、streaming 状态等结构。服务层只负责把外部请求转换成这些字段，不直接管理模型内部的计算对象。

## 生命周期怎么流转

最小生命周期可以先看成五个状态：

1. `waiting`：请求已经进入队列，但还没有开始 prefill。
2. `prefill`：prompt tokens 被送入模型，KV cache 写入完整上下文。
3. `running`：每个 decode step 只推进新 token，并复用历史 cache。
4. `finished`：触发停止条件，结果可以返回。
5. `aborted`：用户取消、资源不足或服务侧主动终止。

这个生命周期解释了为什么 scheduler 不能只保存一个待生成列表。它需要知道每个请求处于哪一步、占了多少 cache、下一步是否还能继续执行。

服务化场景还要补两条规则：

- `aborted` 请求不能再次进入 waiting 或 running 队列。
- 进入 `finished` 或 `aborted` 后，KV Cache、workspace 和流式输出状态都必须进入回收路径。

## 对照 vLLM / nano-vLLM 时看什么

先找请求对象，再找它如何进入等待队列和运行队列。不要一开始就追所有优化策略，先确认三个问题：

- 一个请求什么时候从 waiting 进入 prefill？
- prefill 结束后，它的 KV cache 视图保存在哪里？
- decode 每一步生成 token 后，请求状态在哪里更新？

这三个问题构成 serving runtime 的主线。后面的连续批处理、抢占、block 管理和 streaming 输出，都是在这个生命周期上继续增加规则。

相关阅读：[`推理引擎服务化基础`](../reference/service_basics.md)、[`推理引擎内存管理`](../reference/memory_management.md)
