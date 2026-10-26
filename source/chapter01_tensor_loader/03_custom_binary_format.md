# 01.3 Custom Binary Format 自定义二进制格式


## 明确自定义权重格式的边界

设计一个简单、可解析、可扩展的模型文件格式，用来存放 tensor 元信息和原始权重数据。

读完后应能做到：

- 理解为什么需要自定义格式
- 理解文件格式中元信息和数据区的分离
- 理解魔数、tensor table 和 data 区的作用

## 代码直达

- <a href="../_code/chapter01_tensor_loader/src/model_loader.h"><code>model_loader.h</code></a>
- <a href="../_code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a>

## 格式设计

模型加载器要读的不只是“文件”，而是一种有结构的描述。格式定义清楚后，C++ 侧解析才会稳定，Python 转换脚本才有明确目标。

文件结构建议：

- magic number
- version
- tensor table
- data region

设计原则：

- 结构简单，便于调试
- 元信息和原始数据分离
- 方便顺序读取
- 方便后续增加 dtype、device 或 layout 信息

这个格式不需要一开始就像正式模型文件那样复杂。它更重要的价值是“可演进”：今天存 name、shape、dtype 和数据，明天还能加版本号、layout、量化参数，而不必重新设计整个文件结构。

## 自定义格式在真实推理里的位置

真实项目里的 checkpoint 或权重格式通常会更复杂，但它们都在解决同一个问题：怎样让静态文件能被稳定恢复成运行时对象。

所以本节不是在教一种私有文件格式，而是在建立两个判断：

- 元信息和数据区必须能分开理解
- 版本、shape、dtype 这些 contract 必须先被文件稳定表达

这样后面看真实项目的 safetensors、HF checkpoint 或其他格式时，学习者才不会只看到“读文件”，而能看出“这是在给 runtime 准备对象入口”。

## 验证与排查

验证时重点看：

- 能从文件头判断格式是否正确
- 能读取 tensor 名称、shape 和偏移
- 能定位到每个 tensor 的数据区

只看文件头和 tensor table 就能大致理解文件内容时，这个格式通常就足够清楚了。

常见问题：

- 格式过度设计
- 元信息和数据混在一起
- 版本字段缺失，后续无法演进
