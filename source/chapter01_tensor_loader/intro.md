# 01.0 Load Weights Safely 加载模型权重并统一表示

这一章先回答一个基础问题：模型权重怎样安全地进入程序，并被统一表示成 `Tensor` 和 `ModelLoader` 这套对象模型？同时也会说明“数据加载”和“权重加载”的关系：前者是更一般的输入读取问题，后者是本章要解决的具体场景。

如果你只是想先理解“怎么用”，可以先看本章的源码入口 [`tensor.h`](/code/chapter01_tensor_loader)、[`model_loader.cpp`](/code/chapter01_tensor_loader)，再回到这里看概念和推导。

本章会实现一个可用的 `Tensor` 抽象，并把模型权重从自定义格式加载到程序中。

这一章是主线的起点。后面的 `tensor ops`、`matmul`、`attention` 和 `generation` 都建立在这里定义的 `Tensor` 与加载链路之上。

这一章会碰到的 C++ 概念主要是：类与对象、构造与析构、`shared_ptr`、`vector`、`string`、`unordered_map`，以及头文件和源码分离。

这里最关键的理解是：这一章不只是“把文件读进来”，而是在建立一个可信输入入口。权重一旦进入程序，后面的所有章节都默认它的 shape、dtype、device 和 ownership 已经被正确表达。

```mermaid
flowchart LR
    F[weight file] --> L[ModelLoader]
    L --> T[Tensor]
    T --> P[print / inspect]
    P --> N[chapter02_tensor_ops]
```

## 1. 本章概览

| 先看什么 | 这一章给什么 | 为什么重要 | 对应小节 |
| --- | --- | --- | --- |
| 为什么先做 loader | 可信输入入口 | 后面所有章都依赖它 | [`01.3`](#3-背景是什么) / [`01.5`](#5-中间推导是什么) |
| Tensor 先表达什么 | shape、dtype、device、ownership | 输入语义先固定 | [`01.5`](#5-中间推导是什么) / [`01.6`](#6-本章会产出什么) |
| 怎么验证加载正确 | 名称和形状打印、最小闭环 | 先确认对象模型成立 | [`01.9`](#9-怎么验证) / [`01.10`](#10-本章解决了什么) |
| 源码直达 | `tensor.*` / `model_loader.*` | 先看对象模型和加载逻辑 | [`tensor.h`](/code/chapter01_tensor_loader) / [`model_loader.cpp`](/code/chapter01_tensor_loader) |
| 下一步接到哪里 | `chapter02_tensor_ops` | view 语义建立在这层基础上 | [`01.11`](#11-下一步是什么) |

## 2. 阅读入口

- 前置阅读：[`chapter00_quickstart`](/chapter00_quickstart/intro)、[`foundation`](/foundation/README)
- 源码直达：[`tensor.h`](/code/chapter01_tensor_loader)、[`tensor.cpp`](/code/chapter01_tensor_loader)、[`model_loader.h`](/code/chapter01_tensor_loader)、[`model_loader.cpp`](/code/chapter01_tensor_loader)、[`chapter01_tensor_loader_test.cpp`](/code/chapter01_tensor_loader)
- 相关阅读：[`01.1 Tensor Interface`](/chapter01_tensor_loader/01_tensor_interface)、[`01.5 ModelLoader`](/chapter01_tensor_loader/05_model_loader)、[`01.7 本章小结`](/chapter01_tensor_loader/summary)

## 3. 背景是什么

如果你对面向对象设计不熟，先看 `foundation/05_oop_and_design_basics.md`，重点理解“对象为什么要把状态和行为放在一起”。

如果你对容器、复杂度和查找结构不熟，先看 `foundation/06_data_structure_algorithm_basics.md`，重点理解为什么这里常用 `vector` 和 `unordered_map`。

这不是额外负担，而是为了让你在读 `Tensor` 和 `ModelLoader` 时，不会把“数据结构选择”和“对象设计”同时当成障碍。

如果把这一章写深，重点不只是 loader 的文件格式，而是要说明为什么模型权重必须先被统一装进一个清晰的对象模型里：这样后面的算子、view 语义和 runtime 才能把“数据从哪里来”这件事从计算逻辑中分离出去。

## 4. 最小设计是什么

最小设计不是一次性把所有格式都支持完，而是先建立两层职责：

- `Tensor` 负责把数据表示成清晰对象
- `ModelLoader` 负责把文件内容转成这些对象

## 5. 中间推导是什么

如果从“模型权重如何进入程序”这个问题倒推，会先碰到几个约束：

- 权重不是一次性临时变量，而是后续多章都会复用的输入对象
- 后面要同时支持 shape、dtype、device 和 ownership，所以不能只用简单数组承载
- 如果对象模型不清楚，后面的 `reshape`、`slice` 和算子输入都会很难写
- 如果 loader 只做字节搬运，语义检查就会散落到后续每一章

把这些约束摆出来后，设计自然会收敛成两层：

- `Tensor` 负责把数据表示成清晰对象
- `ModelLoader` 负责把文件内容转成这些对象

这个推导的关键转折点在于：**先定义对象，再定义读取方式**。  
也就是说，我们不是先有文件格式再去适配对象，而是先把程序里需要怎样表示权重说清楚，再把外部文件映射进来。

## 6. 本章会产出什么

- 一个能承载权重数据的 `Tensor`
- 一个可扩展的模型文件格式思路
- 一个能把文件内容转成内存对象的 `ModelLoader`
- 一个最小可验证的权重读取闭环

## 7. 实现路径是什么

1. 定义 `Tensor` 接口
2. 实现 `Tensor` 的内存管理
3. 设计自定义二进制格式
4. 编写 Python 转换脚本
5. 实现 C++ `ModelLoader`
6. 打印并验证加载结果
7. 确认 loader 和 Tensor 的职责边界

## 8. 常见问题是什么

- 以为 loader 只负责读字节，不负责表达语义
- 以为 shape 对了就代表张量对象设计已经对了
- 以为模型文件格式可以随便变，只要能反序列化就行
- 以为 ownership 和 lifetime 是后面的优化问题，而不是加载阶段的基础约束

## 9. 怎么验证

- Tensor 能表达 shape、dtype、device 和数据指针
- 模型权重能从文件进入程序
- 程序能打印出加载到的 tensor 名称和形状

## 10. 本章解决了什么

这一章把“模型状态如何进入程序”这道门固定住了。  
后面的 view 语义、算子输入和生成 runtime，都会在这套对象模型上继续往下走。

## 11. 下一步是什么

进入 `chapter02_tensor_ops`，在已经可用的 `Tensor` 基础上增加 `reshape` 和 `slice` 这种 view 语义。

## 12. 回看源码

- 先看 [`tensor.h`](/code/chapter01_tensor_loader) 和 [`model_loader.h`](/code/chapter01_tensor_loader)
- 再看 [`tensor.cpp`](/code/chapter01_tensor_loader) 和 [`model_loader.cpp`](/code/chapter01_tensor_loader)
- 最后看 [`chapter01_tensor_loader_test.cpp`](/code/chapter01_tensor_loader)
