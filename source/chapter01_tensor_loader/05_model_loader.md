# 01.5 ModelLoader 模型加载器


## 把文件格式变成模型对象

实现 C++ `ModelLoader`，把自定义格式文件解析成程序内部的 `Tensor` 集合。

读完后应能做到：

- 理解 loader 的输入、输出和生命周期
- 理解如何从文件读取 tensor table 和 data region
- 理解如何把文件内容组织成可访问的内存对象

## 代码直达

- <a href="../_code/chapter01_tensor_loader/src/model_loader.h"><code>model_loader.h</code></a>
- <a href="../_code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a>
- <a href="../_code/chapter01_tensor_loader/tests/chapter01_tensor_loader_test.cpp"><code>chapter01_tensor_loader_test.cpp</code></a>

## Loader 设计

ModelLoader 把“文件格式”真正转成“程序可用数据”。没有 loader，格式只是文档；有了 loader，后面的算子才能吃到真实权重。

核心职责：

- 打开模型文件
- 校验魔数和版本
- 读取 tensor 元信息
- 构建 tensor map
- 提供按名称访问 tensor 的能力

设计要点：

- 错误处理要清晰
- 读取逻辑要可扩展
- 由 Loader 对象持有解析状态和错误信息
- 保留后续支持量化格式的空间

`ModelLoader` 的工作不是“把文件读出来”这么简单，而是把文件里分散的元信息和数据区域组织成可以按名称访问的内存对象。换句话说，它在做的是把静态文件转换成运行时结构。

这也是为什么错误处理要尽量清晰：一旦读错了，后面每个算子都可能连带出问题。

## ModelLoader 为什么是关键入口

如果只选本章一个最像真实推理入口链路的页面，就是 ModelLoader。因为它正好处在：

- 文件格式已经定义完
- 权重也已经能被导出
- 但后面的 runtime 还拿不到可用对象

`ModelLoader` 真正负责的，就是把“静态描述”变成“运行时可访问对象”。这也是读 vLLM / nano-vLLM 时最应该优先寻找的一类模块：权重是在什么地方真正被接成 tensor map、name lookup 或 model state 的。

所以 ModelLoader 建立的不是普通 I/O 习惯，而是一条 runtime 入口判断：

- 哪一步还只是文件
- 哪一步已经是对象
- 哪一步开始可以被后续算子按名称访问

## 验证与排查

验证时重点看：

- 能成功加载完整模型文件
- 能按名称找到 tensor
- 能打印 tensor 的 shape 和 dtype

能够通过名称快速定位某个权重，并确认它的元信息与文件内容一致，loader 基本就已经完成使命了。

常见问题：

- 文件头读取错误
- tensor 偏移计算错误
- 名称和数据映射不一致
