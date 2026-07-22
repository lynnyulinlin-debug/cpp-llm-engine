# 06.7 TensorRT-LLM / Gemma / vLLM 对照：把量化放回真实生态里看

到这里，你已经把量化链路接进了一条稳定的生成 runtime。  
这一节不再重复讲 INT8、反量化和精度验证，而是把本章这条链路放到 `TensorRT-LLM`、`Gemma` 和 `vLLM` 里对照着看，帮你分清：

- 哪些是量化链路本身必须保住的边界
- 哪些是高性能 engine 里的量化做法
- 哪些是模型家族和 serving runtime 会真正影响到的地方

## 1. 本节概览

| 本章里的东西 | `TensorRT-LLM` 里会看什么 | `Gemma` 里会看什么 | `vLLM` 里会看什么 |
| --- | --- | --- | --- |
| 量化权重 | engine 级量化和部署路径 | 具体模型家族的兼容性和精度表现 | 量化后 runtime 的吞吐和服务行为 |
| `DequantizeOp` | 算子融合和执行路径 | 模型结构如何影响还原误差 | 运行时里如何保持 cache 和调度稳定 |
| 量化 Linear | 高性能 GPU engine 中的算子实现 | 量化后还能不能保持可接受误差 | 量化后还能不能继续服务高并发场景 |
| 精度验证 | 部署前的工程判断 | 模型家族作为验证对象 | serving 场景下的回归边界 |

## 代码直达

- [`src/quantized_linear.cpp`](/code/chapter06_quantization#quantized-linear-cpp)
- [`src/dequantize_op.cpp`](/code/chapter06_quantization#dequantize-op-cpp)
- [`src/tensor.cpp`](/code/chapter06_quantization#tensor-cpp)
- [`tests/chapter06_quantization_test.cpp`](/code/chapter06_quantization#chapter06-quantization-test-cpp)
- [`examples/main_quantization.cpp`](/code/chapter06_quantization#main-quantization-cpp)

## 2. 为什么它重要

本章已经把量化做成了一条清楚的链路：

- 权重先变成更小的表示
- loader 再把它读进程序
- `DequantizeOp` 把它送回可计算状态
- `Linear` 和后续算子继续沿着原来的 runtime 跑
- 精度验证负责告诉你这条路值不值得保留

`TensorRT-LLM`、`Gemma` 和 `vLLM` 分别对应这条链路里最常遇到的三个问题：

1. engine 怎么接量化
2. 模型家族会不会放大量化误差
3. 量化后的 runtime 还能不能继续服务

所以这节的作用，不是再补一遍背景，而是把本章已经做出来的量化路径放回真实生态里看它会落到哪里。

## 3. 先看什么

阅读顺序如下：

1. 先看本章的量化权重和 `DequantizeOp`
2. 再看 `TensorRT-LLM` 怎么把量化放进 engine 和部署路径
3. 再看 `Gemma` 会怎样影响形状、兼容性和误差判断
4. 最后看 `vLLM` 怎么把量化后的 runtime 继续跑成高吞吐服务

这四步里，前两步最接近“量化怎么做”，后两步更像“量化做完以后会碰到什么”。

## 4. 中间推导是什么

往回推这三者为什么都要看，通常会得到几个约束：

- 量化不能只停留在文件层，必须进到 runtime
- 反量化不能是黑盒，必须能说清楚误差从哪里来
- 模型家族不同，量化后能接受的误差边界就可能不同
- 高吞吐 serving 里，量化不能破坏调度和 cache 的稳定性
- 高性能 engine 里，量化会变成部署策略的一部分，而不是单独的文件格式

把这些约束摆出来后，就能看懂为什么本章和这三类生态是同一条主线上的不同落点：

- 本章先把量化链路做清楚
- `TensorRT-LLM` 看的是高性能 engine 怎么接这条链路
- `Gemma` 看的是模型家族会把这条链路放大成什么问题
- `vLLM` 看的是量化后的 runtime 还能不能继续稳定服务

## 5. 学习后应能做什么

本节完成后，应当能够把下面几件事说清楚：

- 为什么量化权重不能只看成“更小的文件”
- 为什么 `DequantizeOp` 必须和 loader、Tensor dtype、Linear 一起看
- 为什么 `Gemma` 更适合拿来观察量化后的误差和兼容性
- 为什么 `vLLM` 更适合拿来观察量化后的运行时行为

把这些对应关系讲明白，就说明你已经不只是知道量化怎么实现，而是知道量化在真实生态里会被谁接住、被谁验证、被谁继续跑下去。

## 6. 边界说明

这一节不负责复刻 `TensorRT-LLM`、`Gemma` 或 `vLLM` 的全部能力。

它只负责把本章最小量化链路和这三个生态的关注点对齐，让你知道：

- 哪些概念已经在本章里出现
- 哪些设计只是把这些概念做得更完整
- 哪些地方是工程落点，不是量化本身

## 7. 参考页

- [TensorRT-LLM](/reference/ecosystem/tensorrt_llm)
- [Gemma](/reference/ecosystem/gemma)
- [vLLM](/reference/ecosystem/vllm)
- [chapter06_quantization 代码入口](/code/chapter06_quantization)
