# 00.0 Get Started with a Real Baseline 验证本地推理基线

这一章先回答一个最实际的问题：本地环境怎样最快跑通一次真实推理？

本章只负责把第一条真实链路跑通：准备依赖、编译 `llama.cpp`、下载一个小模型，并完成第一次推理验证。

## 本章概览

| 先看什么 | 这一节讲什么 | 为什么重要 | 对应小节 |
| --- | --- | --- | --- |
| 环境怎么准备 | 基础工具和运行条件 | 先排除环境问题 | [`00.1 Environment Setup`](./01_environment_setup.md) |
| 怎么编译基线 | 克隆并编译 `llama.cpp` | 先得到可运行基线 | [`00.2 Clone and Build`](./02_clone_and_build_llamacpp.md) |
| 模型怎么准备 | 下载 TinyLlama | 先拿到真实输入 | [`00.3 Download TinyLlama`](./03_download_tinyllama.md) |
| 怎么完成首次推理 | 运行第一次本地推理 | 先看到真实输出 | [`00.4 First Inference`](./04_first_inference.md) |
| 出错怎么排查 | 常见环境和路径问题 | 先把起跑线稳定住 | [`00.5 Troubleshooting`](./05_troubleshooting.md) |
| 怎么体验 serving baseline | 可选部署 vLLM OpenAI-compatible server | 先看到真实推理服务形态 | [`00.6 Deploy vLLM`](./06_deploy_vllm.md) |
| 本章怎么收口 | 确认最小基线成立 | 进入主线前先固定状态 | [`00.7 本章小结`](./summary.md) |

## 阅读入口

- 前置阅读：[`foundation`](../foundation/README.md)
- 源码直达：<a href="../_code/chapter00_quickstart/main_quick.cpp"><code>main_quick.cpp</code></a>、<a href="../_code/chapter00_quickstart/run.sh"><code>run.sh</code></a>、<a href="../_code/chapter00_quickstart/CMakeLists.txt"><code>CMakeLists.txt</code></a>
- 相关阅读：[`00.1 Environment Setup`](./01_environment_setup.md)、[`00.4 First Inference`](./04_first_inference.md)、[`00.6 Deploy vLLM`](./06_deploy_vllm.md)、[`00.7 本章小结`](./summary.md)

## 这章在解决什么

这一章只做一件事：把第一条真实推理链路跑通。它不负责教程导航，也不展开讲 C++、CMake 或 CUDA 基础；它要先确认教程所依赖的最小运行环境是成立的。

## 这章怎么做

这一章只保留最小但真实的一条路径：

- 准备依赖
- 编译 `llama.cpp`
- 下载小模型
- 运行第一次推理
- 可选部署一次 `vLLM` serving baseline
- 记录成功时的输出形态

这意味着它不追求覆盖所有运行场景，而是先把“教程真的能跑起来”这件事固定住。`vLLM` 小节是可选体验，用来提前看到真实推理服务的形态，不作为后续章节的强制依赖。

## 收口与下一步

这一章的验证很直接：能成功编译、能下载并加载模型、能看到第一次推理的输出。它把教程的起跑线固定在一个可复现的状态上。

下一步进入 `chapter01_tensor_loader`，开始把模型权重安全地读进程序，并把输入语义固定下来。
