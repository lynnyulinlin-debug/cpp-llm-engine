# 01.6 Load Verification and Printing 加载验证与打印


## 这一页要解决什么

加载模型并打印所有 tensor 的名称和形状，用最小可见结果确认链路正确。

读完后应能做到：

- 知道如何验证模型是否真的被正确加载
- 知道如何通过打印 metadata 做快速排查
- 知道这一页是整个加载链路的最终验收

## 代码直达

- <a href="../_code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a>
- <a href="../_code/chapter01_tensor_loader/tests/chapter01_tensor_loader_test.cpp"><code>chapter01_tensor_loader_test.cpp</code></a>

## 验证内容

如果没有最终验证，前面的格式设计和 loader 实现都只是“看起来对”。这一步是最小闭环验收点。

建议打印内容：

- tensor name
- shape
- dtype
- device
- storage size

验证方式：

- 模型文件能完整读取
- tensor 数量合理
- 每个 tensor 的 shape 能被正确打印
- 没有乱码、空字段或明显错位

打印检查的意义不是替代测试，而是提供最早、最轻量的肉眼验证方式。尤其在 loader 刚完成时，先打印出正确的名称和 shape，往往比直接进入算子调试更有效。

## 这一页在真实工程里对应什么

真实推理项目里，权重加载问题往往不会一开始就表现成“程序崩了”，而更常见的是：

- 名称对不上
- shape 对不上
- 某些 tensor 缺失

所以这一页的价值，是把“最早能发现问题的检查点”固定下来。它说明 loader 完成之后，不要立刻跳进算子或性能问题，先确认最基础的对象恢复结果是不是可信。

常见问题：

- 打印结果和文件内容不一致
- tensor 数量少于预期
- 名称和 shape 对不上

这一页输出能够和文件里的实际内容一一对应时，说明从文件到内存的链路已经闭合了。
