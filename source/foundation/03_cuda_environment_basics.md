# F03 CUDA and Environment Basics CUDA 和环境基础

## 目标

理解这套教程所需的最小 CUDA 环境知识。

## 读完你应该能做什么

- 知道这套项目里 CUDA 是用来做什么的
- 从高层理解 GPU、driver、toolkit 和 runtime
- 理解什么时候教程可以用 CPU fallback
- 理解基本环境变量和设备检查

## 为什么重要

后面的章节会把 CUDA 当成矩阵乘法和模型执行的加速路径。要理解这条路径，读者需要先知道哪些东西在 driver 里，哪些东西在 toolkit 里，以及 GPU 不可用时程序还能做什么。

这一章刻意不展开厂商细节。目标是让环境可理解，而不是把它写成一份完整 CUDA 手册。

## 后面会用到什么

- 基于 cuBLAS 的 MatMul
- GPU 加速推理
- CUDA 编译和排错
- 什么时候 CPU fallback 有用

## 本教程里的 CPU 和 CUDA

CPU 路径已经足够完成这些工作：

- 读代码并理解教程结构
- 构建和运行主学习路径
- 验证 tensor shape、数据流和正确性检查
- 跑前面那些更偏概念、所有权、布局和算子边界的章节

CUDA 则在你需要这些事情时才变成必需：

- benchmark GPU 加速的 matmul 或 attention 实现
- 编译 `chapter03` 里显式启用 CUDA backend 的部分
- 对比 CPU reference 和 GPU backend
- 使用 cuBLAS 这类 CUDA 库的加速路径

换句话说：CPU 足够学习和验证正确性，CUDA 主要服务于加速和后端比较。

如果你想先看常见 CUDA 库和用途的简表，可以看 [07_cuda_common_libraries.md](./07_cuda_common_libraries.md)。这一章只聚焦环境和运行时检查。

## 这一页的核心内容

### 1. Driver 和 Toolkit

driver 负责让系统和 GPU 通信，toolkit 提供编译器、头文件和 CUDA 代码需要的库。

### 2. GPU 内存和设备选择

模型权重和 activation 会落在 device memory 里，选中的设备必须和构建/运行时假设一致。

### 3. CUDA 构建开关

构建开关决定项目走 CPU-only 还是 CUDA-enabled 路径。

在这套教程里，`llama.cpp` 这类外部参考项目会用它们自己的选项名，比如 `GGML_CUDA=ON`；而我们自己的教程示例则更倾向于用 `ENABLE_CUDA` 这种更通用的名字。读者要做的不是背选项名，而是看懂“构建开关”和“代码路径”之间的关系。

### 4. 基本运行时检查

在开始排查模型失败之前，先确认设备是否可见、运行时是否真的能使用它。

这里最实用的判断不是只看 `nvidia-smi`，而是看程序能不能真的枚举到设备并完成一次最小 runtime 检查。`nvidia-smi` 说明驱动层大概率可见，`nvcc --version` 说明编译器在，但真正能不能跑，要看 runtime。

```cpp
#include <cuda_runtime.h>
#include <iostream>

bool check_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::cerr << "CUDA not available: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    return true;
}
```

如果你安装了 CUDA samples，`deviceQuery` 也是一个很好的参考工具，因为它更接近“runtime 能不能真正用上 GPU”这个问题。这里不要求读者先会它，但它是一个值得知道的名字。

在多卡机器上，`CUDA_VISIBLE_DEVICES` 也很有用：

```bash
CUDA_VISIBLE_DEVICES=0 ./my_program
CUDA_VISIBLE_DEVICES="" ./my_program
```

前者用于限定只看一张卡，后者用于强制走 CPU fallback，适合隔离环境问题。

## 可以一直保持 CPU-first 的章节

下面这些部分是故意设计成 CPU-friendly 的，不需要把 CUDA 当成硬前提：

- `chapter00_quickstart`
- `chapter01_tensor_loader`
- `chapter02_tensor_ops`
- `chapter04_attention`
- `chapter03_matmul` 的 CPU reference 路径
- `chapter05_generation`
- `chapter06_quantization`
- `chapter07_benchmark`

真正需要 CUDA 的方向，出现在你想比较 backend 或把 GPU 加速当作优化路径的时候，而不是学习门槛本身。对 `chapter03_matmul` 来说，这意味着 CPU reference 可以先用 CPU-first 路径读完，只有 `cuBLAS` 后端和 GPU 对照需要 CUDA 环境。

## 最小示例

```bash
# 示例检查，按本机环境调整
nvidia-smi
nvcc --version

# 构建思路：CPU-only vs CUDA-enabled
cmake -B build -DCMAKE_BUILD_TYPE=Release
# 或者
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
```

这个例子说明：

- 机器能看到 GPU
- CUDA 编译器已经安装
- 构建路径可以显式切换
- 只看 `nvidia-smi` 不够，还要能做 runtime 检查

`GGML_CUDA=ON` 是外部参考项目 `llama.cpp` 的写法；如果是在我们自己的教程代码里，读者应当期待的是类似 `ENABLE_CUDA=ON` 的通用命名，而不是固定记住某个第三方选项。

## 怎么验证

- 确认系统能看到 GPU
- 确认当前构建路径和想要的 backend 一致
- 确认在没有 CUDA 的情况下，教程仍然可以用 CPU 运行
- 确认程序本身能枚举到设备，而不是只靠命令行工具判断
- 确认 `CUDA_VISIBLE_DEVICES` 改变后，程序行为也跟着变化

## 常见问题

- 把 driver 和 toolkit 混淆
- CUDA 构建好了，但运行机器上缺少需要的 runtime
- 忘了有些工作流需要 CPU fallback 作为基线
- 把环境配置当成和使用该环境的代码路径无关
- 一开始就假设每一章都必须有 CUDA
- 等 GPU 环境“完全完美”才去验证 CPU 路径
- 只用 `nvidia-smi` 判断一切正常
- 把构建开关当成纯粹的命令行细节，而不是代码路径的一部分

## 环境排查

| 现象 | 可能原因 | 检查方法 |
| --- | --- | --- |
| `nvidia-smi` 有输出，但程序报 CUDA 错误 | Driver 版本或 runtime 不匹配 | 对照 driver、runtime 和运行时错误信息 |
| CUDA 编译成功，运行时失败 | Toolkit 和 runtime 路径不一致 | 对比 `nvcc --version` 和程序实际运行环境 |
| 多卡环境只用到一张卡 | `CUDA_VISIBLE_DEVICES` 没有按预期设置 | 检查环境变量和设备枚举结果 |
| 只能看到 GPU，但跑不通 CUDA kernel | 只完成了“可见”，没有完成“可用” | 用 `cudaGetDeviceCount` 和 `deviceQuery` 再确认一次 |

## 总结

这一章解释了后面章节会用到的 CUDA 背景。读完后，读者应该知道在把问题归咎于模型或代码之前，先检查什么。

如果你能解释 driver、toolkit 和 runtime 的区别，并且能判断问题属于 CPU 逻辑还是 CUDA 配置，那么这层环境知识就已经足够支撑后面的教程了。
