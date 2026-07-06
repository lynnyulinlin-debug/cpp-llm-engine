# chapter01_tensor_loader 代码索引

## 权重加载与 Tensor

- `code/chapter01_tensor_loader/src/tensor.cpp`
- `code/chapter01_tensor_loader/src/tensor.h`
- `code/chapter01_tensor_loader/src/model_loader.cpp`
- `code/chapter01_tensor_loader/src/model_loader.h`

## 测试

- `code/chapter01_tensor_loader/tests/chapter01_tensor_loader_test.cpp`

## 对应文档

- [chapter01 导读](../chapter01_tensor_loader/intro)
- [chapter01 小结](../chapter01_tensor_loader/summary)

## 说明

这一章的代码先建立输入语义，再让 loader 把文件内容稳定转成 Tensor。

## 使用方式

如果你想沿着 chapter01 的文档看代码，建议按这个顺序：

1. 先看 [chapter01 导读](../chapter01_tensor_loader/intro)，理解“数据加载”和“权重加载”的关系
2. 再看 `tensor.cpp` / `tensor.h`，确认输入对象是怎样被表达的
3. 再看 `model_loader.cpp` / `model_loader.h`，确认文件内容是怎样被转换成对象的
4. 最后看测试，确认加载链路和对象语义是稳定的

如果后续章节要复用这一章的能力，优先复用的是：

- `Tensor` 的对象语义
- `ModelLoader` 的输入边界
- 加载验证的最小测试方式
