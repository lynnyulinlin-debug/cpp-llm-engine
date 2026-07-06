# 03.4 cuBLAS Implementation cuBLAS 实现


## 这一页要解决什么

用 `cublasGemmEx` 实现高效矩阵乘法，并正确处理转置标志与内存布局。

## 阅读入口

- 前置阅读：[`MatMulOp 设计`](/chapter03_matmul/02_matmul_design)、[`CPU Reference`](/chapter03_matmul/03_cpu_reference)
- 源码直达：[`chapter03_matmul 代码入口`](/code/chapter03_matmul)
- 相关阅读：[`PyTorch 对照测试`](/chapter03_matmul/05_test_with_pytorch)、[`复杂度直觉`](/chapter03_matmul/06_complexity_intuition)

## 读完你应该能做什么

- 理解 cuBLAS 在本章中的作用
- 理解如何把内部张量布局映射到 cuBLAS 调用
- 理解转置参数和内存布局的关系

## 为什么这一步重要

这一步把“正确的参考实现”升级成“高效实现”。如果布局处理不对，结果会错得很隐蔽，所以这一页必须和 CPU baseline 紧密对照。

## 这一页的核心内容

- `cublasGemmEx`
- `cublasHandle_t`
- `transa` / `transb`
- row-major 与 column-major 的映射

工程实现上，建议把 cuBLAS 调用包在一个很薄的适配层里。适配层只做三件事：

1. 统一输入输出接口
2. 翻译 layout / transpose 参数
3. 管理 handle 和临时资源

这样如果后续想切换其他 GPU backend，不需要重写上层算子逻辑。

### row-major 到 cuBLAS 的映射示例

如果项目内部统一使用 row-major，而你要计算 `C = A * B`，那么在 cuBLAS 里通常会把同一块内存按 column-major 的视角解释成转置后的矩阵来算。

一个常见的等价写法是：

- 内部语义：`C = A * B`
- cuBLAS 视角：计算 `C^T = B^T * A^T`
- 对应设置：`transa = CUBLAS_OP_T`，`transb = CUBLAS_OP_T`

这样做的原因是：cuBLAS 默认按 column-major 理解输入，而项目内部可能按 row-major 存储。  
把这层翻译说清楚之后，读者在看 cuBLAS 文档时就不会把“数学上的转置”和“内存布局上的解释”混为一谈。

## 实现建议

- 先保持接口和 CPU 版本一致
- 只替换底层计算路径
- 保留和 CPU baseline 相同的测试入口

## 怎么验证

- 和 CPU 参考实现数值接近
- 和 PyTorch 对照误差可接受
- 布局转换后输出正确

如果 cuBLAS 版本和 CPU 参考版本在小样本上完全一致，通常说明接口和布局映射已经过了第一道关。

最小适配层可以先长这样：

```cpp
Tensor matmul_cublas(const Tensor& a, const Tensor& b, cublasHandle_t handle) {
    // 1. 校验 shape
    // 2. 翻译 layout / transpose
    // 3. 调用 cublasGemmEx
    // 4. 返回输出 Tensor
}
```

关键不是把 cuBLAS API 直接暴露给上层，而是把它包在一个可替换的 backend 里。

## 常见问题

- 转置标志设置反了
- 行主序和列主序混用
- 输入输出维度与 GEMM 参数不一致
