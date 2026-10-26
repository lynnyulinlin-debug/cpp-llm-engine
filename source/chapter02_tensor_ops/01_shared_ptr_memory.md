# 02.1 Shared Underlying Storage 共享底层存储


## 让多个 Tensor 共享存储

让多个 `Tensor` 对象可以共享同一份底层数据，而不是每次视图变化都复制内存。

读完后应能做到：

- 理解为什么要共享底层存储
- 理解 `shared_ptr` 在 view 语义中的角色
- 理解“数据”和“视图”分离的意义

## 代码直达

- <a href="../_code/chapter02_tensor_ops/include/chapter02_tensor_ops/tensor.h"><code>tensor.h</code></a>
- <a href="../_code/chapter02_tensor_ops/src/tensor.cpp"><code>tensor.cpp</code></a>

## 核心语义

`reshape` 和 `slice` 都不应该随意复制数据。复制不仅浪费内存，还会让视图语义失去意义。  
共享底层存储是实现高效 Tensor 操作的前提。

更具体一点说，核心问题是“如何让一个数据对象同时支持多种观察方式”。
如果没有共享存储，view 语义就只是一个看起来省事、实际上很贵的包装。

最稳妥的做法是把责任拆开：

- 底层数据交给共享所有权对象管理
- `Tensor` 只保存自己的视图信息
- 多个视图可以指向同一份存储

实现上，最直接的做法是让 `Tensor` 持有一个共享存储句柄，并额外保存自己的偏移量和形状信息。这样，复制 `Tensor` 对象只会复制视图描述，不会复制底层字节。

一个典型的行为是：

```cpp
Tensor a = Tensor::from_vector<float>("a", {4}, {1.f, 2.f, 3.f, 4.f});
Tensor b = a;
// b 和 a 指向同一份 storage，但可以拥有不同的 shape / offset 解释
```

这种设计的关键不是“省一次拷贝”这么简单，而是为后面的 `reshape`、`slice`、`KV Cache` 复用同一套存储语义。

## 共享存储在真实推理里的位置

如果说 `chapter01` 解决的是“权重怎样进入程序”，本节解决的就是下一步：进入程序后的数据怎样被多个视图复用。

在真实推理项目里，这种共享存储语义通常会出现在三类地方：

- 输入或中间张量的多种 shape 解释
- attention 或 projection 前后的局部视图
- KV cache 这类需要长期持有、反复观察的数据块

共享存储需要满足一条约束：view 可以多，storage 不能乱复制。

示例伪代码可以写成：

```cpp
struct TensorView {
    std::shared_ptr<std::vector<std::byte>> storage;
    std::size_t offset_bytes;
    std::vector<int64_t> shape;
};

TensorView make_view(const TensorView& parent, std::size_t offset, std::vector<int64_t> shape) {
    return TensorView{parent.storage, offset, std::move(shape)};
}
```

这类结构的重点是“视图”只保存引用，不复制底层字节。

实现建议：

- 使用 `std::shared_ptr<uint8_t>` 或等价存储句柄管理原始数据
- 让 `Tensor` 持有偏移、长度和 shape 信息
- 不把 view 的派生数据单独复制出来

## 验证与排查

验证时重点看：

- 多个视图对象可以同时访问同一份数据
- 一个视图销毁后，底层存储不会被提前释放
- 共享存储不会导致重复释放

如果共享存储设计正确，读者应能观察到：

- 父张量和子张量读到相同的底层内容
- 销毁子张量不会影响父张量的数据可读性
- 多次复制对象不会引发 double free

常见问题：

- 把共享存储误写成独占所有权
- 忘记保存偏移信息
- 让 view 对象和底层数据生命周期不一致
