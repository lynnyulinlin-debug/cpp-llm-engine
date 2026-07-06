# 03.1 Operator Base 算子基类


## 1. 这一页要解决什么

定义所有前向算子都能继承的基础接口，让 `MatMul`、`LayerNorm`、`FFN` 这类模块能用同一种方式接入主流程。

## 阅读入口

- 前置阅读：[`chapter03 导读`](/chapter03_matmul/intro)、[`MatMulOp 设计`](/chapter03_matmul/02_matmul_design)
- 源码直达：[`chapter03_matmul 代码入口`](/code/chapter03_matmul)
- 相关阅读：[`CPU Reference`](/chapter03_matmul/03_cpu_reference)、[`cuBLAS 实现`](/chapter03_matmul/04_cublas_implement)

## 2. 读完你应该能做什么

- 理解为什么算子需要统一接口
- 理解 `forward` 的职责边界
- 理解算子如何和 `Tensor` 协作

## 3. 为什么这一步重要

当算子越来越多时，如果每个算子都用不同的入口形式，主流程就会变成一堆特例。  
统一的基类不是为了“抽象得更漂亮”，而是为了让主线在接入新模块时保持稳定。

## 4. 这个基类应该长什么样

最小版本只需要一件事：定义 `forward`。

这个 `forward` 要尽量窄，只负责“输入什么、输出什么”，不要在基类里塞进过多参数。  
如果某个算子还需要额外状态，应该让子类自己保存，而不是把基类做成万能接口。

最小版本可以先支持这样的调用方式：

```cpp
Tensor y = op.forward(x);
```

源码锚点：`code/chapter03_matmul/include/chapter03_matmul/operator.h`

这个接口默认适合“单输入 -> 单输出”的算子。对于需要多个独立输入的算子，比如后续的 Attention，如果继续保留这个窄接口，调用方就需要先把多个 Tensor 组合成一个结构再传入，或者在后续版本里把接口升级成 `forward(const std::vector<Tensor>&)`。  
这是一种有意的取舍：接口越窄，基类越稳定，但上层组织输入的责任就越大。

后面如果某个算子需要多个输入，再把接口扩展成接收输入张量列表，而不是提前把所有可能性都塞进去。  
这样做的好处是：新算子只需要继承统一接口，主流程不需要为每个模块写新的分支。

## 5. 最小示意

```cpp
class Operator {
public:
    virtual ~Operator() = default;
    virtual Tensor forward(const Tensor& input) = 0;
};
```

## 6. 怎么验证

- 其他算子可以继承该基类
- `forward` 接口可以统一调用
- 基类不承担具体计算逻辑

如果你能把 `MatMulOp`、`LayerNormOp`、`FFNOp` 放进同一套调用流程里，而不需要为每个算子写特殊分支，这个基类就已经起作用了。

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

源码锚点：`code/chapter03_matmul/src/operator.cpp`

这说明基类只负责接口，不负责具体数学逻辑。

## 7. 常见问题

- 基类设计过宽
- 强行把所有算子参数塞进同一个接口
- 让接口依赖过多上下文
