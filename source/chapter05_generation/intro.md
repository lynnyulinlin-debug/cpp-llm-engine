# 05.0 Turn Forward into a Generative Runtime 构建生成运行时

这一章先回答一个系统问题：怎么把 `forward` 变成真正能逐 token 输出的 runtime？

这一章沿用本教程一贯的工程写法：先定问题，再拆边界，最后用测试把链路闭起来。  
它不追求一次讲完所有 serving 技巧，而是先把 runtime 转折点做成一个可验证、可继续演进的最小骨架。

```mermaid
flowchart LR
    P[prompt] --> PF[prefill]
    PF --> KS[KV Cache]
    PF --> D[decode]
    D --> S[sampler]
    S --> O[output token]
    O --> D
    D --> F[finished]
    R[request state] --- PF
    R --- D
    Q[scheduler] --> PF
    Q --> D
```

## 1. 本章概览

| 先看什么 | 这一章给什么 | 为什么重要 | 对应小节 |
| --- | --- | --- | --- |
| 生成为什么会变成 runtime 问题 | `prefill / decode` 的状态机 | 这是从结构走向运行时的分界线 | [`05.2`](#2-本章要解决什么) / [`05.4`](#4-最小设计是什么) / [`05.5`](#5-中间推导是什么) |
| runtime 最先要暴露什么 | 请求状态、cache 状态、调度状态 | 没有这三层，后面很难做优化判断 | [`05.4`](#4-最小设计是什么) / [`05.5`](#5-中间推导是什么) |
| 最小实现先解决什么 | naive generation + 最小 scheduler + KV Cache | 先把“能生成”做对，再谈吞吐 | [`05.4`](#4-最小设计是什么) / [`05.6`](#6-本章会产出什么？) / [`05.8`](#8-怎么验证？) |
| 代码入口 | `站内代码索引` | 先把文档和代码连起来 | [`代码索引`](/code/chapter05_generation) |
| 这一章怎么收口 | 端到端生成测试 + 后续量化 / benchmark 桥接 | 让后续章节继续复用同一条 runtime 主线 | [`05.8`](#8-怎么验证？) / [`05.10`](#10-本章解决了什么？) / [`05.11`](#11-下一步是什么？) |

## 2. 本章要解决什么

`chapter04` 已经把单层 decoder 做稳了，但还没有把它放进生成循环。这一章要做的，就是把 decoder 接到 runtime 里，做出一条可重复运行的最小生成链路。

一旦进入生成场景，问题就会从“结构是否站得住”变成“结构如何进入 runtime”。  
后面的量化和 benchmark 只有建立在稳定 runtime 上，才有可比的基线。

## 3. 背景是什么

生成式模型和一次性前向计算最大的区别，不在于“有没有输出”，而在于“输出是不是一 token 一 token 地长出来”。  
只要进入生成场景，系统就会多出三个新问题：请求状态要怎么保存、缓存要怎么复用、调度要怎么推进。

这一章放在 decoder 之后，是因为 decoder 只能说明“单层结构能工作”，还不能说明“多个请求如何在同一 runtime 里被持续服务”。  
所以这里的背景不是“再讲一遍 Transformer”，而是把结构问题推进到运行时问题。

## 4. 最小设计是什么

这一章先把最小 runtime 骨架搭出来：

- 请求进入后先走 `prefill`
- `prefill` 结束后进入 `decode`
- 每个请求都有自己的状态
- 每个请求都有自己的 cache 视图
- 调度器只负责决定下一步推进谁

这意味着本章先不追求复杂 serving，而是先把最小但可观察的生成状态机做出来。

## 5. 中间推导是什么

如果从“怎样把 `forward` 变成 runtime”倒推，通常会得到下面的约束：

- 输出必须是逐 token 的，而不是一次性把整段结果算完
- 请求状态必须显式保存，否则多请求场景无法区分彼此
- `prefill` 和 `decode` 必须分开理解，否则缓存边界会模糊
- KV Cache 必须尽早出现，否则重复计算会快速放大
- 调度器必须能看到活跃请求，否则 runtime 只能处理单一路径

把这些约束摆出来后，设计就会自然收敛成一条状态机链路：`prompt` 进入、`prefill` 建缓存、`decode` 逐 token 推进、`finished` 收尾。  
这个推导的关键点不是“加更多模块”，而是把生成从一次性函数调用改造成可持续运行的状态流转。

## 6. 本章会产出什么

- embedding 和位置编码模块
- N 层 Transformer 组装
- naive generation 实现
- KV Cache 设计与实现
- sampler 和端到端生成测试

这些产出共同服务于一个目标：把“模型前向”变成一条可观察的运行时链路。

## 7. 实现路径是什么

1. Embedding 和位置编码
2. N 层 Transformer 组装
3. naive generation
4. KV Cache
5. sampler
6. 端到端生成测试
7. 请求状态机与调度边界

## 8. 怎么验证

- 先确认单条 prompt 能得到稳定输出
- 再确认加入 KV Cache 后输出行为仍然正确
- 最后确认端到端链路能够重复运行并产生日志或文本结果

## 9. 常见问题是什么

- 把 generation 当成一次性 forward，忽略了状态流转
- 只按单请求写代码，结果一旦出现多个请求就互相污染
- `prefill` 和 `decode` 没有分开，导致 cache 边界模糊
- cache 只会增长不会被观察，最后内存压力说不清楚
- scheduler 只是隐式地选一个请求，但没有把推进规则显式化

## 10. 本章解决了什么

这一章把 `chapter04` 已经站稳的单层 decoder 接到生成流程里，做出一条可重复运行的最小 runtime：`prefill` 先建状态，`decode` 再逐 token 推进，`Request`、`KV Cache` 和 `Scheduler` 全部显式化。

## 11. 下一步是什么

- `chapter06_quantization`：换掉权重表示和计算路径，但保留 runtime 组织方式
- `chapter07_benchmark`：把 runtime 放到统一测试和性能框架里，验证它是否真的可比较、可回归、可优化

## 12. English anchors

- `Problem`: move from forward pass to generation runtime
- `Why it matters`: runtime state becomes visible
- `Minimal design`: `prefill / decode` + `Request / KV Cache / Scheduler`
- `Verification`: token-by-token generation and end-to-end test
- `Next step`: quantized inference path and benchmark evidence chain

## 13. 相关阅读

- 前置依赖：建议先读 [chapter04_attention 导读](/chapter04_attention/intro) 和 [chapter04_attention 小结](/chapter04_attention/summary)，因为本章的生成 runtime 直接建立在 DecoderLayer 结构上。
- 后续使用：`chapter06_quantization` 会保留这一章的 runtime 组织方式，只替换权重表示和计算路径；`chapter07_benchmark` 会在这条 runtime 上做统一测量。
- 路径说明：如果想先看 03–06 的整体路径，可以直接参考 [chapter03_matmul 导读](/chapter03_matmul/intro)，它已经汇总了从算子到量化的阅读顺序。
