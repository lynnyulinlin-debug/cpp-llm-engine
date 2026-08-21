# 03.2 MatMulOp Design `MatMulOp` 设计


## 这一页要解决什么

这一页要把 `MatMulOp` 的边界先定下来。读完后，读者应该能说清楚三件事：

- 这个算子接收什么输入
- 它输出什么结果
- CPU reference 和 cuBLAS 为什么是同一个接口下的两种实现

矩阵乘法是后面很多模块的公共底座。如果这一层设计不清楚，attention 和 FFN 的问题会和布局问题混在一起，排错会很慢。

## 阅读入口

- 前置阅读：[`Operator Base`](./01_operator_base.md)、[`CPU Reference`](./03_cpu_reference.md)
- 源码直达：<a href="../_code/chapter03_matmul/include/chapter03_matmul/matmul_op.h"><code>matmul_op.h</code></a>、<a href="../_code/chapter03_matmul/src/matmul_op.cpp"><code>matmul_op.cpp</code></a>
- 相关阅读：[`cuBLAS 实现`](./04_cublas_implement.md)、[`PyTorch 对照测试`](./05_test_with_pytorch.md)

## 核心设计

在推理引擎里，矩阵乘法不是“先写一个公式就完了”的东西。真正难的地方是把内部 Tensor 布局、安全的资源管理，以及外部高性能库的调用方式接起来。

如果直接从 cuBLAS 调用开始写，读者很容易只看到参数，没有看到设计约束。  
如果先把接口说清楚，再讨论后端实现，就能把“语义”与“性能”分开。

`MatMulOp` 不是一个单纯的计算函数，而是一个稳定边界：

- 上层只依赖它的输入输出契约
- 下层可以切换 CPU reference 或 cuBLAS
- 布局转换只发生在边界内，不向上层泄漏

这里最容易出错的不是矩阵乘法本身，而是内部 layout 到库调用参数的映射。  
因此设计阶段必须先明确项目内部使用什么布局，再决定如何翻译成 cuBLAS 所需的参数。

`cublasHandle_t` 的生命周期要独立管理，不能让它散落在调用点。  
同时，传给 cuBLAS 的矩阵布局可能和项目内部布局不同，所以转置标志、leading dimension 和内存顺序必须一起考虑。

如果内部统一采用 row-major，很多时候要把输入视作转置后的 column-major 形式交给 cuBLAS。  
这不是实现细节，而是设计约束。

这里的接口约定也要明确：`forward(A, B)` 表示 `A * B`，也就是左乘语义。  
只要这个语义在 CPU 和 cuBLAS 后端之间保持一致，上层在拼接 Q/K/V 或组织 FFN 输入时就不会因为参数顺序而写反。

## 为什么这一层必须单独抽出来

如果把矩阵乘法直接埋进模型代码里，后面很快会出现三个问题：

- 上层模型语义和 layout 细节混在一起
- backend 切换会扩散到每个调用点
- attention、FFN、projection 的排错边界会变得很模糊

所以 `MatMulOp` 这一层的真正作用，不只是“包装一次乘法”，而是把职责切开：

- 上层只说“我要一次什么形状、什么语义的乘法”
- backend 层决定“这次乘法由谁来算”
- kernel 或库调用层负责“具体怎么高效地算”

这也是读 vLLM / nano-vLLM 时最容易混淆的一点。很多人一上来就看 kernel，但真正应该先看的是：这里到底是在定义算子语义，还是在替换 backend 实现。

## 验证与排查

验证时重点看：

读完后，应当能够不看实现就回答：

- 输入是什么
- 输出是什么
- 布局怎么转换
- 资源怎么释放
- 为什么先保证正确性，再谈加速

可以先用一个简化调用把思路写出来：

```cpp
Tensor y = matmul_op.forward(x, w);
// 内部再根据 backend 选择 CPU reference 或 cuBLAS 路径
```

源码锚点：`code/chapter03_matmul/include/chapter03_matmul/matmul_op.h`

这个例子不是为了展示代码量，而是为了强调：前端接口要稳定，后端执行要可替换。

常见问题：

- 只考虑数学公式，不考虑内存布局
- 把 cuBLAS 输入布局和内部布局混为一谈
- handle 没有明确释放路径
