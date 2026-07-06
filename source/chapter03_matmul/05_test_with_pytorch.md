# 03.5 PyTorch Comparison Testing PyTorch 对照测试


## 这一页要解决什么

随机生成矩阵，与 PyTorch 的结果对照，验证 `MatMul` 的数值正确性。

## 阅读入口

- 前置阅读：[`CPU Reference`](/chapter03_matmul/03_cpu_reference)、[`cuBLAS 实现`](/chapter03_matmul/04_cublas_implement)
- 源码直达：[`chapter03_matmul 代码入口`](/code/chapter03_matmul)
- 相关阅读：[`复杂度直觉`](/chapter03_matmul/06_complexity_intuition)、[`本章小结`](/chapter03_matmul/summary)

## 读完你应该能做什么

- 理解为什么要做对照测试
- 理解数值误差应该怎么判断
- 理解 correctness test 和 performance test 的区别

## 为什么这一步重要

矩阵乘法是很多后续模块的基础。一旦这里出现偏差，attention 和 FFN 的误差会被放大，所以必须有稳定的对照测试。

## 测试策略

- 随机生成小矩阵
- 分别跑 CPU 版本和 cuBLAS 版本
- 与 PyTorch 结果比较
- 检查误差是否在阈值内

建议把随机测试固定随机种子，这样一旦失败，输入可以复现。对于工程教程来说，可复现比“随机覆盖更多”更重要。

### 测试复用关系

这套 PyTorch 对照测试可以直接复用 `03.3` 和 `03.4` 的测试入口：

- `test_matmul_cpu`
- `test_matmul_cublas`

两者使用相同的随机输入，再分别与 PyTorch 对照。

这样就能形成一条清晰的双基线路径：

- 如果 CPU 版本通过、cuBLAS 版本失败，问题大概率在布局翻译或 backend 参数
- 如果 CPU 和 cuBLAS 都通过，说明基线可信
- 如果两者都失败，问题大概率在数学定义或测试输入本身

这样做的好处是，读者可以沿着同一组输入同时检查参考实现、GPU 实现和 PyTorch 基线，而不是为三者分别维护一套不同的测试逻辑。

## 验证标准

- 输出 shape 一致
- 数值误差足够小
- 边界输入也能被正确处理

如果 PyTorch 对照测试能稳定区分“实现错误”和“数值误差”，这个测试就有实际价值，而不是形式上的单测。

一个最小对照伪代码可以是：

```python
torch.manual_seed(0)
A = torch.randn(2, 3)
B = torch.randn(3, 4)
golden = A @ B
```

然后把同样的输入送到本实现中，比较输出和 `golden` 的差异。

## 常见问题

- 误差阈值设得过严
- PyTorch 输入布局和本实现不一致
- 随机测试只覆盖了部分情况
