# F09 Minimal C++/CUDA Project 构建一个最小的 C++/CUDA 项目

## 目标

先让读者理解最小可用的 C++ 项目应该长什么样，再理解如何在不推翻结构的情况下把它扩展成 CUDA-enabled 项目。

## 读完你应该能做什么

- 理解最小 C++ 项目的目录布局
- 理解 `main.cpp`、头文件、源文件和 `CMakeLists.txt` 如何配合
- 理解如何先加一个 library target，再加 CUDA
- 理解如何从 CPU-only 构建切换到 CUDA 构建

## 为什么重要

很多初学者还没开始写真正代码，就被项目布局劝退了。这套教程想做相反的事情：先从一个文件、一个 target、一个 executable 开始，等确实需要时再慢慢增长。

这一章是“我会 C++ 语法”到“我能构建并运行一个真实引擎项目”之间的桥。

关键点是：第一个能跑的项目应该足够小，便于理解；同时又要足够稳定，便于继续长大。如果这里的结构清楚，后面的章节模块就能更容易加进来，而不用重写整棵目录树。

## 后面会用到什么

- 教程自己的 `code/chapterXX_*` 项目
- CPU 和 CUDA 构建路径
- library target 和 test target
- 在不破坏构建的前提下新增一个源文件

## 与 F02 的关系

`F02` 介绍了 `project()`、`add_executable()`、`add_library()`、`target_link_libraries()` 等命令。
`F09` 则把这些命令组合成一个真实可运行的最小项目。

如果你对某个 CMake 命令不熟，可以回到 `F02` 看对应说明；如果你更关心“这些命令最终怎么拼起来”，就看这一页。

## 这一页的核心内容

### 1. 先从一个 executable 开始

最小有用项目可以只是一个 `main.cpp` 和一个 executable target。这样你可以先有一个能跑的构建结果，再继续往里加东西。

这个“第一次成功”很重要，因为它把“工具链问题”和“项目设计问题”分开了。

### 2. 先拆成 library 和 app

一旦 executable 能跑，就把可复用逻辑移到 library target。这样项目可以增长，但不会变成一个巨大的单文件。

对这套教程来说，这就是从小 demo 走向可复用章节模块的正常路径。

### 3. 头文件只在需要时增加

声明放头文件，实现放源文件。保持这种分离小而清楚。

教程就是按这个规则写的，这样读者能看清接口在哪里结束，实现在哪里开始。

目录边界也可以随项目增长自然演进：

```text
project/
├── include/        # 公开接口，给其他模块用
│   └── core.h
├── src/            # 实现文件
│   ├── core.cpp
│   └── kernel.cu
└── app/            # 可执行入口
    └── main.cpp
```

当前示例先用一个更小的 `src/` 版本起步是可以的，但当项目开始增长时，`include/` 和 `src/` 分离会更清楚。

### 4. 把 CUDA 当成 target 扩展，而不是惊喜

CPU 版本先工作，再加一个 CUDA 文件和一个构建开关。这样 CUDA 是同一个项目的扩展，而不是另一个 puzzle。

这样 GPU 支持就变成可控增长，而不是第一天就藏着一个第二工程。

最小的 CUDA 文件可以长这样：

```cpp
// src/kernel.cu
#include <cuda_runtime.h>

__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

这不是要在这一页讲 CUDA kernel 细节，而是让读者知道：`.cu` 文件本身看起来和 `.cpp` 文件很像，只是编译目标不同。

## 最小示例

### 目录结构

```text
mini_project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── core.cpp
│   ├── core.h
│   └── kernel.cu
```

### 只有 CPU 的版本

```cmake
cmake_minimum_required(VERSION 3.20)
project(mini_project LANGUAGES CXX)

add_library(core src/core.cpp)
target_compile_features(core PUBLIC cxx_std_17)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(app src/main.cpp)
target_link_libraries(app PRIVATE core)
```

```cpp
// src/core.h
#pragma once

int add(int a, int b);
```

```cpp
// src/core.cpp
#include "core.h"

int add(int a, int b) {
    return a + b;
}
```

```cpp
// src/main.cpp
#include "core.h"
#include <iostream>

int main() {
    std::cout << add(1, 2) << "\n";
    return 0;
}
```

### CUDA 扩展

```cmake
option(ENABLE_CUDA "Enable CUDA support" OFF)

if(ENABLE_CUDA)
    enable_language(CUDA)
    add_library(cuda_kernels src/kernel.cu)
    target_link_libraries(app PRIVATE cuda_kernels)
endif()
```

这样可以保证，即使机器上没有安装 CUDA，CPU 构建也仍然工作。

从单文件到多文件的演进，可以粗略理解成：

```text
# Step 1: 单文件
+ src/main.cpp (包含所有代码)

# Step 2: 拆出 library
+ src/core.cpp
+ src/core.h
- src/main.cpp (只保留 main 函数)

# Step 3: 添加 CUDA
+ src/kernel.cu
+ CMakeLists.txt (添加 ENABLE_CUDA 选项)
```

## 怎么验证

- 能配置并构建 CPU-only 项目
- 能运行 executable 并看到预期输出
- 能在不改整体布局的情况下新增一个源文件
- 能打开 CUDA 选项，同时仍然保持清晰的构建路径
- 能分别构建 CPU-only 和 CUDA-enabled 版本
- 能对比两个版本的构建日志，理解差异在哪里

如果这些检查都通过，读者就应该能理解项目是怎样增长的，而不会失去构建 CPU 基线的能力。

## 常见问题

- 什么都写在一个文件里
- CPU 路径都还不稳定，就先加 CUDA
- 忘了区分接口和实现
- 构建系统依赖全局状态，而不是 target-local 设置

## 总结

读完这一章后，读者应该知道一个很小的 C++ 项目如何以受控方式长成一个支持 CUDA 的项目。这就是主教程采用的构建模式。

如果你能把 CPU 版本干净地构建出来，先加一个可复用 library，然后再把 CUDA 当成 option 而不是重写，那项目结构对后面的教程就已经足够了。
