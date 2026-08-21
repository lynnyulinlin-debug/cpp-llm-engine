# 00.4 First Inference 第一次推理

## 阅读入口

- 前置阅读：[`00.2 Clone and Build`](./02_clone_and_build_llamacpp.md)、[`00.3 Download TinyLlama`](./03_download_tinyllama.md)
- 源码直达：<a href="../_code/chapter00_quickstart/main_quick.cpp"><code>main_quick.cpp</code></a>、<a href="../_code/chapter00_quickstart/run.sh"><code>run.sh</code></a>
- 相关阅读：[`00.5 Troubleshooting`](./05_troubleshooting.md)、[`推理引擎输入链路`](../reference/input_pipeline.md)、[`00.7 本章小结`](./summary.md)

## 这一页要解决什么

运行 TinyLlama，观察第一次推理输出，并确认整个链路已经跑通。

## 读完你应该能做什么

- 知道如何启动模型推理
- 知道如何判断输出是否正常
- 知道 CPU 路径可以作为最稳妥的回退方案

## 前置条件

- 已完成编译
- 已下载模型
- 知道 `llama.cpp` 的可执行文件在哪里

## 步骤

### CPU 路径

```bash
cd /data/cpp-llm-engine/llama.cpp/build
./bin/llama-cli -m ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hello, world" -n 1
```

### CUDA 路径

如果你已经编译了 CUDA 版本，并且知道对应的运行参数，可以在同一模型上做 GPU 验证。

## 你应该看到什么

- 程序正常启动
- 模型加载成功
- 控制台输出一段生成结果

示例输出会因模型版本和随机性而变化，但应当是类似自然语言文本，而不是报错信息。

这一页的重点不是输出质量，而是确认命令行输入、模型加载、前向推理和 token 输出构成一条可重复的 baseline。输入契约的完整拆分见 [`推理引擎输入链路`](../reference/input_pipeline.md)。

## 怎么验证

- 命令执行完毕且退出码正常
- 模型输出不是空的
- 没有路径错误、加载错误或显存错误

如果第一次推理能稳定返回，后面的章节就不需要再怀疑环境本身。此时可以把注意力转到自研引擎的抽象和实现上。

## 常见问题

- 模型路径写错
- 可执行文件路径写错
- CPU 版本和 CUDA 版本混用
- 输出太慢但不是错误
