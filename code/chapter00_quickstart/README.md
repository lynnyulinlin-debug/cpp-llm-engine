# Chapter 00：Quickstart 真实推理起点

本目录验证外部真实推理基线，不实现教程自己的 Tensor、Decoder 或 Runtime。完整学习说明见 [`source/chapter00_quickstart`](../../source/chapter00_quickstart/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `CMakeLists.txt` | 构建本章 smoke executable |
| `main_quick.cpp` | 检查模型路径和 quickstart 入口 |
| `run.sh` | 从临时目录配置、构建并运行 smoke |
| `llama.cpp/` | 外部真实推理参考，不属于本章源码资产 |
| `llama.cpp/models/*.gguf` | TinyLlama 等真实模型文件，不提交到仓库 |

## 本章代码范围

- 检查 CPU/CUDA 前置环境
- 编译 `llama.cpp`
- 下载和检查 TinyLlama GGUF
- 使用真实 tokenizer 完成第一次推理
- 将真实 baseline 与教程自研实现分开

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter00_quickstart/run.sh
```

该命令只验证本仓库的 smoke executable。真实模型推理使用 `llama.cpp/build/bin/llama-simple`，命令见 Chapter 00 正文。
