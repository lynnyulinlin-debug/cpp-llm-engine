# 02.2 Reshape 重塑


## 这一页要解决什么

实现只改变形状、不复制数据的 `reshape` 操作。

## 读完后应能做什么

- 理解 `reshape` 的语义边界
- 理解为什么 `reshape` 不能偷偷拷贝数据
- 理解 `reshape` 对后续算子的意义

## 代码直达

- [`include/chapter02_tensor_ops/tensor.h`](/code/chapter02_tensor_ops#tensor-h)
- [`src/tensor.cpp`](/code/chapter02_tensor_ops#tensor-cpp)
- [`tests/chapter02_tensor_ops_test.cpp`](/code/chapter02_tensor_ops#chapter02-tensor-ops-test-cpp)

## 为什么这一步重要

`reshape` 是最典型的 view 操作之一。它只改变张量的形状解释，不改变底层存储内容。

## 语义要求

- 只改变 `shape`
- 不复制底层数据
- 保持数据总元素数一致
- 保持底层存储句柄不变

`reshape` 的关键约束是“元素总数必须不变”。如果新旧 shape 的元素数不同，这个操作就不再是视图变换，而是在请求一份不同的数据布局，应该直接拒绝。

最小实现通常只做三件事：

1. 校验新 shape 的元素数
2. 复用原始存储和偏移
3. 返回一个只更新 shape 的新视图

示例伪代码可以写成：

```cpp
Tensor reshape(const Tensor& t, std::vector<int64_t> new_shape) {
    if (numel(new_shape) != t.numel()) {
        throw std::invalid_argument("reshape size mismatch");
    }
    Tensor out = t;
    out.set_shape(std::move(new_shape));
    return out;
}
```

这里最重要的是不要分配新存储，只改视图描述。

## 实现建议

- 检查元素数量是否匹配
- 复用原始存储
- 返回新的张量视图对象

## 怎么验证

- `reshape` 前后元素总数一致
- 数据内容不变
- 新旧张量共享同一份存储

一个简单的验证方式是把 `{2, 2}` reshape 成 `{4}`，再把结果转回 vector 比较，确认内容顺序没有变化。

## 常见问题

- reshape 后偷偷分配了新内存
- shape 修改后元素数量不一致
- 忘记保留原始 storage
