# 01.1 Tensor Interface Design Tensor 接口设计


## 这一页要解决什么

定义后续所有算子和模型加载都会使用的 `Tensor` 基础接口。

读完后应能做到：

- 明确 `Tensor` 需要包含哪些核心字段
- 理解 `shape`、`dtype`、`device` 和数据存储之间的关系
- 理解为什么 `Tensor` 要先作为抽象存在，而不是直接操作裸指针

## 代码直达

- <a href="../_code/chapter01_tensor_loader/src/tensor.h"><code>tensor.h</code></a>
- <a href="../_code/chapter01_tensor_loader/src/tensor.cpp"><code>tensor.cpp</code></a>

## 核心设计

后面的算子、模型加载、量化和缓存管理，都依赖统一的张量抽象。  
这一层设计不清楚时，后续章节就会不停重复处理内存、形状和设备这些基础问题。

换句话说，这一页不是在定义“一个更大的容器”，而是在定义整个教程共享的数据语言。  
一旦 `Tensor` 的边界不稳定，后面的每一章都会被迫重新解释同样的概念。

`Tensor` 至少要表达四类信息：

- `shape`
- `dtype`
- `device`
- data pointer / storage handle

这四项里，`shape` 和 `storage` 最容易被混在一起。实际上，`shape` 描述的是逻辑视图，`storage` 描述的是物理数据。  
把这两层分开，后面的 `reshape`、`slice` 和缓存语义才有机会成立。

`Tensor` 的责任要克制：

- 只负责表达数据，不把算子逻辑塞进去
- 既要能描述元信息，也要能指向真实存储
- 形状信息和存储信息要分开管理
- 后续应支持 view 语义，为 `reshape` 和 `slice` 做准备

这一层的设计要尽量克制。`Tensor` 不是“计算一切的对象”，而是“统一表达数据的容器”。  
只要这层边界清楚，后面的加载、算子、量化和缓存都能沿着同一个抽象继续扩展。

特别要注意的是，`shape` 描述的是逻辑视图，`storage` 描述的是物理数据。  
把这两层混在一起，后续的 view 语义就会变得很难做。

最小接口示意：

```cpp
struct Tensor {
    std::vector<int64_t> shape;
    DType dtype;
    Device device;
    std::shared_ptr<void> storage;
    std::size_t offset_bytes = 0;
};
```

## 这一页在真实推理里对应什么

如果说 loader 负责把文件内容读进来，这一页负责的就是下一步：这些内容一旦进入程序，要用什么对象语义继续往后传。

在真实推理项目里，这种 `Tensor` 语义通常会同时服务三类东西：

- 权重对象本身
- 后续算子输入输出
- runtime 里需要继续携带的 metadata

所以这一页真正建立的是一条底线：后面的章节不应该反复重新定义“shape / dtype / device / storage 到底是什么意思”。

## 验证与排查

验证时重点看：

- 能说清每个字段的作用
- 能区分“张量元信息”和“底层存储”
- 能解释为什么后续需要 shared storage

能够在不看实现的情况下解释“为什么 `Tensor` 需要 shape、dtype、device 和 storage 四类信息”，说明这一页的抽象已经成立。

常见问题：

- 只存 shape，不存 dtype
- 把 device 信息散落到各个算子里
- 让 Tensor 直接承担过多行为
