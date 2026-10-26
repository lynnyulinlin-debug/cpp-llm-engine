# 01.2 Teaching Weight Format 教学权重格式

## 学习目标

本章用很小的 `CIE1` 格式展示“磁盘字节如何恢复成 Tensor”。它是 loader 契约的测试夹具，真实 TinyLlama 路径使用 GGUF。读完后应能：

- 找出文件头、tensor record 和 payload
- 说明导出端与加载端必须共享哪些字段
- 判断错误来自格式、元数据还是 payload

代码入口：

- <a href="../_code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a>
- <a href="../_code/chapter01_tensor_loader/tests/chapter01_tensor_loader_test.cpp"><code>chapter01_tensor_loader_test.cpp</code></a>

## CIE1 文件布局

测试中的 fixture writer 按以下顺序写入数据，`ModelLoader` 按相同顺序读取：

| 区域 | 字段 | 作用 |
| --- | --- | --- |
| 文件头 | `magic=CIE1`、version、tensor count | 识别格式并选择解析规则 |
| Tensor record | name、dtype、device、ndim、shape | 恢复 Tensor 元数据 |
| Payload | byte count、raw bytes | 恢复实际权重数据 |

这个最小格式把关键契约显式摆出来：导出端写入的字段顺序、整数宽度和 dtype 编号，必须与 C++ loader 完全一致。测试 fixture 直接在 C++ 中生成文件，因此可以同时固定成功路径和截断、错误 magic 等失败路径。

训练生态通常用 Python 读取 Hugging Face checkpoint，再写入目标格式。若为本教程补充 converter，它应复用这份格式契约，并验证 tensor 数量、名称、shape、dtype 和 payload 字节数；它不参与 C++ runtime。当前真实模型不经过该转换，而是直接读取 GGUF。

## 从教学格式到 GGUF

两条路径最终都产生内部 Tensor，但复杂度不同：

| 教学路径 | 真实路径 |
| --- | --- |
| fixture writer → CIE1 → `ModelLoader` | TinyLlama GGUF → `GgufReader` → adapter / `GgufTensorLoader` |
| 连续 tensor record，字段很少 | metadata、tensor index、alignment 和量化类型 |
| 用于理解格式与错误处理 | 用于真实 tokenizer、权重和 Decoder 验证 |

## 验证

运行 `bash code/chapter01_tensor_loader/run.sh`，确认合法 fixture 能恢复两个 Tensor；再检查错误 magic、错误版本和截断 payload 均返回明确错误。完成后继续阅读 [`01.3 ModelLoader and Validation`](./03_model_loader_validation.md)，观察磁盘记录如何成为可查询对象。

