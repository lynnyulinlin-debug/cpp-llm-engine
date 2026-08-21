# 01.0 Model Weights and Tensor Objects 加载模型权重并统一表示

这一章先回答一个基础问题：模型权重怎样安全地进入程序，并被统一表示成 `Tensor` 和 `ModelLoader` 这套对象模型？

本章会实现一个可用的 `Tensor` 抽象，并把模型权重从自定义格式加载到程序中。

## 本章和 vLLM / nano-vLLM 的联系

在 vLLM 或 nano-vLLM 里，模型首先也要经过配置解析、权重加载、tensor metadata 校验和 shape contract 对齐。本章不直接实现这些项目的 loader，而是先建立最小的 `Tensor` 和 `ModelLoader` 语义，让后面读真实项目时能看懂“权重如何进入 runtime”这条入口链路。

更具体地说，本章处在真实推理执行链的入口层：

- 文件或外部权重先被描述成可解析格式
- loader 再把这些静态内容转成运行时对象
- 后面的 view、算子、attention 和 runtime 都只能建立在这套对象语义上继续往下走

学习本章时，最重要的不是把 `Tensor` 和 `ModelLoader` 当成独立组件，而是先建立一个判断：这一步解决的是“权重怎样进入程序”，而不是“权重进来以后怎样参与计算”。

## 本章概览

| 先看什么 | 这一节讲什么 | 为什么重要 | 对应小节 |
| --- | --- | --- | --- |
| `Tensor` 先表达什么 | shape、dtype、device 和 ownership | 先定输入对象语义 | [`01.1 Tensor Interface`](./01_tensor_interface.md) |
| `Tensor` 怎么落到代码里 | 存储、构造和对象生命周期 | 先把对象边界写稳 | [`01.2 Tensor Implement`](./02_tensor_implement.md) |
| 文件格式怎么设计 | 自定义二进制格式 | 先让权重能被稳定描述 | [`01.3 Custom Binary Format`](./03_custom_binary_format.md) |
| 训练侧权重怎么导出 | Python converter | 先把训练侧和 C++ 侧接起来 | [`01.4 Python Converter`](./04_python_converter.md) |
| C++ 怎么读回权重 | `ModelLoader` | 先把文件还原成运行时对象 | [`01.5 ModelLoader`](./05_model_loader.md) |
| 怎么验证加载结果 | 打印名称和形状 | 先确认加载可信 | [`01.6 Verify and Print`](./06_verify_and_print.md) |
| 本章怎么收口 | 输入语义和下一章入口 | 进入 view 语义前先固定加载链路 | [`01.7 本章小结`](./summary.md) |

## 阅读入口

- 前置阅读：[`chapter00_quickstart`](../chapter00_quickstart/intro.md)、[`foundation`](../foundation/README.md)
- 源码直达：<a href="../_code/chapter01_tensor_loader/src/tensor.h"><code>tensor.h</code></a>、<a href="../_code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a>、<a href="../_code/chapter01_tensor_loader/tests/chapter01_tensor_loader_test.cpp"><code>chapter01_tensor_loader_test.cpp</code></a>
- 相关阅读：[`01.1 Tensor Interface`](./01_tensor_interface.md)、[`01.5 ModelLoader`](./05_model_loader.md)、[`推理引擎输入链路`](../reference/input_pipeline.md)、[`01.7 本章小结`](./summary.md)

## 这章在解决什么

这一章只做一件事：把模型权重安全地读进程序，并把输入语义固定住。它要把 `Tensor` 从“数据容器”推进到“运行时对象”，让后面的算子、view 语义和 runtime 都建立在同一套输入约定上。权重这一侧先定住，prompt、tokenizer 和 stop criteria 再在另一层展开。

它解决的不是“怎么写一个文件读取器”这么局部的问题。权重对象由本章固定，用户请求则由输入链路参考页固定，二者在 `chapter05` 的 `Request` 中汇合。后面真实推理链路因此能够同时做到：

- 权重名称、shape、dtype 有稳定语义
- 文件内容可以按名称和 metadata 被恢复成对象
- 后面的算子和 runtime 不必重新猜测输入对象是什么

## 这章怎么做

最小设计不是一次性把所有格式都支持完，而是先建立两层职责：

- `Tensor` 负责把数据表示成清晰对象
- `ModelLoader` 负责把文件内容转成这些对象

这样一来，数据结构选择和对象设计就不会和加载流程缠在一起。

## 收口与下一步

这一章留下的结果很直接：你知道文件里有什么、loader 读到了什么、最后打印出来的 tensor 名称和形状为什么可信。它把“模型状态如何进入程序”这道门固定住了，后面的 view 语义、算子输入和生成 runtime 都会在这套对象模型上继续往下走。

下一步进入 `chapter02_tensor_ops`，在已经可用的 `Tensor` 基础上增加 `reshape` 和 `slice` 这种 view 语义。
