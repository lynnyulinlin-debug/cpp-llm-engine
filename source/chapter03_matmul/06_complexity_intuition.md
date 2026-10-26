# 03.6 Complexity Intuition 复杂度直觉


## 看懂 MatMul 的复杂度

本节说明为什么 `MatMul` 的参考实现先写成朴素循环，再把性能问题交给 backend。

读完后应能做到：

- 理解矩阵乘法为什么天然就是高计算量操作
- 理解朴素循环为什么适合作为 correctness baseline
- 理解复杂度直觉和工程实现之间的关系

## 阅读入口

- 前置阅读：[`CPU Reference`](./03_cpu_reference.md)、[`cuBLAS 接入设计`](./04_cublas_implement.md)
- 源码直达：<a href="../_code/chapter03_matmul/src/matrix.cpp"><code>matrix.cpp</code></a>、<a href="../_code/chapter03_matmul/src/matmul_op.cpp"><code>matmul_op.cpp</code></a>
- 相关阅读：[`PyTorch 对照测试`](./05_test_with_pytorch.md)、[`从 MatMul 到真实推理 backend`](./07_runtime_backend_bridge.md)

## 核心直觉

很多初学者会想：既然 `MatMul` 很重要，为什么不一开始就把它写得很快？答案是，工程上“先正确，再加速”通常比“先优化，再验证”更稳。

一开始就引入复杂优化时，往往会同时面对数值正确性、内存布局、边界条件和性能调优四类问题。朴素循环把这些问题拆开了：先确认公式和索引对不对，再考虑缓存友好、向量化或 GPU 加速。

- 矩阵乘法的基本工作量来自三层循环
- 小矩阵上，朴素循环更容易和手算结果对照
- 大矩阵上，真正的加速来自 backend，而不是参考实现本身

一个形状为 `M x K` 和 `K x N` 的矩阵乘法，基础计算量大致与 `M * N * K` 成正比。

这意味着：

- 参考实现写得再“聪明”，也不可能从复杂度层面改变本质
- 参考实现的任务是表达正确性，而不是争夺性能
- 后续优化应该集中在更合适的 backend 上

复杂度直觉回答的是“这个计算本身需要多少工作量”；实际运行时间还取决于内存访问模式、cache 命中率、向量化和后端实现。朴素循环负责前者，cuBLAS 负责后者。

最小示例：

```cpp
for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
        float sum = 0.f;
        for (std::size_t k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
```

这段代码的价值是直观：

- 索引关系清楚
- 与数学定义一致
- 容易排查转置和维度错误

## 验证与排查

验证时重点看：

- 能解释为什么三重循环对应矩阵乘法
- 能解释为什么它适合做正确性基线
- 能解释为什么优化应该放到 cuBLAS 或其他 backend

常见问题：

- 把参考实现当成最终性能实现
- 过早优化，导致难以验证
- 忽略矩阵维度和索引的对应关系

复杂度小结：

复杂度小结补充的是性能判断依据。读完后，读者应该能说明为什么 `MatMul` 的 CPU 参考版本先用朴素循环，再把性能交给更合适的后端。
