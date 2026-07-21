# 04.6 Gemma / TensorRT-LLM 对照：把 decoder 结构放回生态语境

到这里，你已经把单层 decoder 做成了一个可验证模块。  
这一节不再重复讲结构拼接，而是把它放到 `Gemma` 和 `TensorRT-LLM` 的语境里对照着看，帮你分清：

- 哪些是 decoder 结构本身必须保住的边界
- 哪些是模型家族带来的结构差异
- 哪些是高性能 engine 进一步拆出来的执行边界

## 1. 本节概览

| 本章里的东西 | `Gemma` 里的对应关注点 | `TensorRT-LLM` 里的对应关注点 | 读的时候重点看什么 |
| --- | --- | --- | --- |
| `DecoderLayer` | 具体模型家族里的 decoder block 结构 | engine 中的 decoder 分层与组装方式 | 结构是怎么闭合的 |
| Attention | Q/K/V、mask 和 residual 的结构约束 | attention kernel / runtime 的实现边界 | 哪些约束不能改 |
| LayerNorm 和 FFN | 模型结构里的标准组件组合 | 算子融合与 runtime 执行路径 | 模块如何被组合起来 |
| 首层权重接入 | 模型家族如何影响 shape 和兼容性 | 权重如何进入 engine 和部署链路 | 权重从哪里接进来 |

## 2. 为什么它重要

本章解决的是“单层 decoder 是否站得住”。  
`Gemma` 和 `TensorRT-LLM` 提醒你，站得住只是第一步，后面还要面对两个现实问题：

1. 不同模型家族的 decoder 结构细节并不完全一样
2. 真正的高性能引擎会把这些结构拆成更细的 runtime 边界

所以这一节不是额外背景，而是把本章的结构结果放回真实生态里的两个对照点：

- `Gemma` 给你看模型家族视角
- `TensorRT-LLM` 给你看高性能 engine 视角

## 3. 先看什么

如果你要先建立一个最小心智模型，可以按这个顺序看：

1. 先看本章的 `DecoderLayer` 和首层权重接入
2. 再看 `Gemma` 里的 decoder block 是怎么组织的
3. 最后看 `TensorRT-LLM` 怎么把 decoder 结构拆进 engine / runtime 分层

这里最重要的不是逐项对照名称，而是看它们如何组织结构、边界和执行路径。

## 4. 中间推导是什么

往回推为什么要把这两类生态放进来，通常会得到几个约束：

- decoder 不能只看“能跑”，还要看结构是否闭合
- 模型家族不同，结构细节就会跟着变化
- engine 不会直接照抄结构，而是会把它拆成更细的执行边界
- 权重接入的位置一变，shape 和兼容性判断也会跟着变

把这些约束摆出来后，就能看懂为什么 `chapter04` 和这两类生态是同一条主线上的不同落点：

- 本章先把单层 decoder 做稳
- `Gemma` 看的是同一结构在具体模型家族里的样子
- `TensorRT-LLM` 看的是同一结构在高性能 engine 里的切分方式

## 5. 你现在应该能做什么

学完本节后，你应该已经能把下面几件事说清楚：

- 为什么单层 decoder 先要闭合，后面才能谈多层堆叠
- 为什么 `Gemma` 更适合拿来观察模型家族层面的结构差异
- 为什么 `TensorRT-LLM` 更适合拿来观察 engine / runtime 的边界
- 为什么首层权重接入不是“加载成功就够了”，还要看结构是否对齐

把这些对应关系讲明白，就说明你已经能把本章的结构放回真实生态里去看。

## 6. 边界说明

这一节不负责复刻 `Gemma` 或 `TensorRT-LLM` 的全部能力。

它只负责把本章的单层 decoder 和这两个生态的关注点对齐，让你知道：

- 哪些概念已经在本章里出现
- 哪些设计只是把这些概念做得更完整
- 哪些地方是工程落点，不是结构本身

## 7. 参考页

- [Gemma](/reference/ecosystem/gemma)
- [TensorRT-LLM](/reference/ecosystem/tensorrt_llm)
- [chapter04_attention 代码入口](/code/chapter04_attention)
