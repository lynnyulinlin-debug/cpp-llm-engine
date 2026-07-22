# 01.4 Python Converter Python 转换脚本


## 这一页要解决什么

把 Hugging Face 上的模型权重转换成本章定义的二进制格式。

## 读完后应能做什么

- 理解转换脚本在整体链路中的位置
- 理解如何把权重、名称和 shape 写入自定义格式
- 理解 Python 转换脚本和 C++ 加载器如何配合

## 代码直达

- [`src/model_loader.h`](/code/chapter01_tensor_loader#model-loader-h)
- [`src/model_loader.cpp`](/code/chapter01_tensor_loader#model-loader-cpp)
- [`tests/chapter01_tensor_loader_test.cpp`](/code/chapter01_tensor_loader#chapter01-tensor-loader-test-cpp)

## 为什么这一步重要

如果没有转换脚本，C++ 侧只能读格式，不能接入真实模型权重。转换脚本是连接训练生态和推理引擎的桥梁。

## 转换脚本职责

- 读取 Hugging Face 模型参数
- 提取 tensor 名称、shape 和数据
- 按自定义格式写入文件
- 为后续 C++ loader 提供稳定输入

这一步的职责是“翻译”而不是“理解模型”。Python 脚本负责把训练生态里的权重转成推理引擎能消费的格式，C++ loader 负责在运行时高效、稳定地读取它们。

把转换逻辑和运行时逻辑分开，后面调试会轻很多。

## 怎么验证

- 生成的 `.bin` 或同类文件可以被后续 C++ 代码读取
- tensor 数量和原始模型一致
- 关键 tensor 的 shape 和名称正确

转换脚本生成的文件能够被 `ModelLoader` 直接接住时，说明训练侧和推理侧之间已经有了明确接口。

## 常见问题

- dtype 没有对齐
- tensor 顺序和 loader 预期不一致
- 转换脚本和文件格式定义不同步
