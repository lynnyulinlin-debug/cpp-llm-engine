# 01.3 ModelLoader and Validation 模型加载与验证

## 学习目标

`ModelLoader` 把 CIE1 文件恢复成按名称查询的 Tensor 集合。本节把加载过程和验收放在一起，读完后应能：

- 追踪 header、metadata 和 payload 的读取顺序
- 用 `find()` 和 `tensor_names()` 检查加载结果
- 为格式错误、缺失权重和 shape 不一致选择最早检查点

代码入口：

- <a href="../_code/chapter01_tensor_loader/src/model_loader.h"><code>model_loader.h</code></a>
- <a href="../_code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a>
- <a href="../_code/chapter01_tensor_loader/tests/chapter01_tensor_loader_test.cpp"><code>chapter01_tensor_loader_test.cpp</code></a>

## 加载过程

`load()` 每次先清空旧状态，再依次校验 magic、version 和 tensor count。每条 record 恢复名称、dtype、device、shape 和 payload，构造 Tensor 后放入 `unordered_map<string, Tensor>`。

加载完成表示文件已经变成运行时可访问对象，但还需要检查对象是否符合模型 schema。排查顺序应从最早的确定性证据开始：

1. 文件能否打开，magic 和 version 是否匹配。
2. tensor record 是否完整，payload 是否截断。
3. 名称、shape、dtype 和 byte count 是否一致。
4. 模型 schema 是否能找到必需权重。
5. 算子输入是否符合已验证的对象契约。

这个顺序可以避免把加载错误带进 Attention 或 MatMul 后再定位。

## 可见结果与自动测试

打印 tensor name、shape、dtype 和 storage size 适合快速观察；断言负责固定长期行为。测试至少覆盖：

- 合法文件恢复完整 tensor map
- 按名称命中已知 tensor，未知名称返回空
- shape 与 payload 可还原为预期数值
- 错误 magic、version 和截断输入被拒绝

真实 GGUF 使用同样的验证思想，但由 [`01.5 Model Metadata and GGUF`](./05_model_metadata_and_gguf.md) 继续检查 metadata、tensor index、alignment、量化 payload 和 Llama schema。

## 验证

```bash
bash code/chapter01_tensor_loader/run.sh
```

通过标准是 CTest 全部成功，并且失败用例在 loader 阶段返回可定位的错误。不要用后续算子的数值结果替代 loader 的名称、shape 和 payload 检查。

