# 01.6 Load Verification and Printing 加载验证与打印


## 这一页要解决什么

加载模型并打印所有 tensor 的名称和形状，用最小可见结果确认链路正确。

## 读完后应能做什么

- 知道如何验证模型是否真的被正确加载
- 知道如何通过打印 metadata 做快速排查
- 知道这一页是整个加载链路的最终验收

## 代码直达

- [`src/model_loader.cpp`](/code/chapter01_tensor_loader#model-loader-cpp)
- [`src/model_loader.h`](/code/chapter01_tensor_loader#model-loader-h)
- [`tests/chapter01_tensor_loader_test.cpp`](/code/chapter01_tensor_loader#chapter01-tensor-loader-test-cpp)

## 为什么这一步重要

如果没有最终验证，前面的格式设计和 loader 实现都只是“看起来对”。这一步是最小闭环验收点。

## 建议打印内容

- tensor name
- shape
- dtype
- device
- storage size

## 验证方式

- 模型文件能完整读取
- tensor 数量合理
- 每个 tensor 的 shape 能被正确打印
- 没有乱码、空字段或明显错位

打印检查的意义不是替代测试，而是提供最早、最轻量的肉眼验证方式。尤其在 loader 刚完成时，先打印出正确的名称和 shape，往往比直接进入算子调试更有效。

## 常见问题

- 打印结果和文件内容不一致
- tensor 数量少于预期
- 名称和 shape 对不上

这一页输出能够和文件里的实际内容一一对应时，说明从文件到内存的链路已经闭合了。
