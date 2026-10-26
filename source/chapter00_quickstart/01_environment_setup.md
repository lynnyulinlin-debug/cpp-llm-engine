# 00.1 Environment Setup 环境准备

**阅读入口**

- 前置阅读：[`chapter00 导读`](./intro.md)、[`foundation`](../foundation/README.md)
- 源码直达：<a href="../_code/chapter00_quickstart/main_quick.cpp"><code>main_quick.cpp</code></a>、<a href="../_code/chapter00_quickstart/run.sh"><code>run.sh</code></a>
- 相关阅读：[`00.2 Clone and Build`](./02_clone_and_build_llamacpp.md)、[`00.4 First Inference`](./04_first_inference.md)

## 准备和目标

准备好本章需要的系统环境、编译工具链和基础依赖。

**完成后：**

- 知道需要哪些基础工具
- 知道什么时候可以先用 CPU 路径验证
- 知道 CUDA 只是可选加速路径，不是快速开始的前置条件

## 环境检查

- Ubuntu 或其他兼容 Linux 发行版
- `git`
- `cmake`
- C++ 编译器
- 可选：CUDA toolkit 和 NVIDIA 驱动

**检查项目：**

- 编译器可用
- `cmake` 可用
- `git` 可用
- 如果要走 GPU 路径，CUDA 相关组件可用

先确认最小可用链路存在。只要 CPU 编译链路可用，就已经能支撑后续大部分教程验证；CUDA 留到后续优化阶段。

如果你同时有 CPU 和 GPU 环境，建议先把 CPU 路径跑通，再回头检查 CUDA。这样一旦后面出问题，你更容易判断是工具链问题还是 GPU 配置问题。

## 怎么验证

- `cmake --version`
- `g++ --version` 或 `clang++ --version`
- `git --version`

如果这些命令都能正常返回版本号，说明最基础的构建工具已经到位。后续只要再确认磁盘空间和网络可用，就可以进入源码编译阶段。

## 常见问题

- 编译器版本太旧
- `cmake` 不在 PATH
- CUDA 驱动和 toolkit 混淆
