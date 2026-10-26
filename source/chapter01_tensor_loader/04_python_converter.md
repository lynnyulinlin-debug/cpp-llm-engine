# 01.4 Python Converter Python 转换脚本


## 把训练权重转换成运行时输入

把 Hugging Face 上的模型权重转换成本章定义的二进制格式。

读完后应能做到：

- 理解转换脚本在整体链路中的位置
- 理解如何把权重、名称和 shape 写入自定义格式
- 理解 Python 转换脚本和 C++ 加载器如何配合

## 代码直达

- <a href="../_code/chapter01_tensor_loader/src/model_loader.h"><code>model_loader.h</code></a>
- <a href="../_code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a>

## 转换链路

如果没有转换脚本，C++ 侧只能读格式，不能接入真实模型权重。转换脚本是连接训练生态和推理引擎的桥梁。

转换脚本职责：

- 读取 Hugging Face 模型参数
- 提取 tensor 名称、shape 和数据
- 按自定义格式写入文件
- 为后续 C++ loader 提供稳定输入

这一步的职责是“翻译”而不是“理解模型”。Python 脚本负责把训练生态里的权重转成推理引擎能消费的格式，C++ loader 负责在运行时高效、稳定地读取它们。

把转换逻辑和运行时逻辑分开，后面调试会轻很多。

## 转换脚本在真实推理里的位置

最值得学习者带走的，不是某个脚本细节，而是训练侧和推理侧之间必须有一条清晰边界：

- 训练生态负责导出权重
- 中间格式负责稳定表达这些权重
- 推理 runtime 负责消费这种表达

读真实项目时，很多“模型支持”问题最后都会落到这里：到底是导出链路的问题，还是运行时 loader 的问题。

## 验证与排查

验证时重点看：

- 生成的 `.bin` 或同类文件可以被后续 C++ 代码读取
- tensor 数量和原始模型一致
- 关键 tensor 的 shape 和名称正确

转换脚本生成的文件能够被 `ModelLoader` 直接接住时，说明训练侧和推理侧之间已经有了明确接口。

常见问题：

- dtype 没有对齐
- tensor 顺序和 loader 预期不一致
- 转换脚本和文件格式定义不同步
