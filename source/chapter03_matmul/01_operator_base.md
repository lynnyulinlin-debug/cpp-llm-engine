# 03.1 Operator Base 算子基类


## 定义统一 Operator 入口

定义所有前向算子都能继承的基础接口，让本章的 `MatMul` 能先用同一种方式接入主流程。

读完后应能做到：

- 理解为什么算子需要统一接口
- 理解 `forward` 的输入输出契约
- 理解算子如何和 `Tensor` 协作

## 阅读入口

- 前置阅读：[`chapter03 导读`](./intro.md)、[`MatMulOp 设计`](./02_matmul_design.md)
- 源码直达：<a href="../_code/chapter03_matmul/include/chapter03_matmul/operator.h"><code>operator.h</code></a>、<a href="../_code/chapter03_matmul/src/operator.cpp"><code>operator.cpp</code></a>
- 相关阅读：[`CPU Reference`](./03_cpu_reference.md)、[`cuBLAS 接入设计`](./04_cublas_implement.md)

## 核心设计

当算子越来越多时，如果每个算子都用不同的入口形式，主流程就会变成一堆特例。  
统一基类让调用方通过稳定接口接入不同算子实现。

最小版本只需要一件事：定义 `forward`。

这个 `forward` 要尽量窄，只负责“输入什么、输出什么”，不要在基类里塞进过多参数。  
如果某个算子还需要额外状态，应该让子类自己保存，而不是把基类做成万能接口。

最小版本可以先支持这样的调用方式：

```cpp
Tensor y = op.forward(x);
```

源码锚点：<a href="../_code/chapter03_matmul/include/chapter03_matmul/operator.h"><code>operator.h</code></a>

这个接口默认适合“单输入 -> 单输出”的算子。对当前这章来说，这已经足够支撑 `MatMul` 的最小边界。
后续如果遇到多输入算子，再决定是组合输入结构，还是把接口升级成 `forward(const std::vector<Tensor>&)`。这是一种有意的取舍：接口越窄，基类越稳定，但上层组织输入的责任就越大。

后面如果某个算子需要多个输入，再把接口扩展成接收输入张量列表，而不是提前把所有可能性都塞进去。  
这样做的好处是：新算子只需要继承统一接口，主流程不需要为每个模块写新的分支。

最小示意：

```cpp
class Operator {
public:
    virtual ~Operator() = default;
    virtual Tensor forward(const Tensor& input) = 0;
};
```

## 验证与排查

验证时重点看：

- 其他算子可以继承该基类
- `forward` 接口可以统一调用
- 基类不承担具体计算逻辑

如果 `MatMulOp` 能放进统一调用流程里，而不需要为这一个核心算子单独写特殊分支，这个基类就已经起作用了。后续再把这种接口稳定性复用到更多算子上。

```cpp
class Operator {
public:
    virtual ~Operator() = default;
    virtual Tensor forward(const Tensor& input) = 0;
};

class DummyOp : public Operator {
public:
    Tensor forward(const Tensor& input) override { return input; }
};
```

源码锚点：<a href="../_code/chapter03_matmul/src/operator.cpp"><code>operator.cpp</code></a>

这说明基类只负责接口，不负责具体数学逻辑。

常见问题：

- 基类设计过宽
- 强行把所有算子参数塞进同一个接口
- 让接口依赖过多上下文
