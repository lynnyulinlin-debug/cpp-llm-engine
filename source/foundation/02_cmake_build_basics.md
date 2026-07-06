# F02 CMake and Build Basics CMake 和编译基础

## 目标

理解这套教程项目是如何配置和构建的。

## 读完你应该能做什么

- 理解 `CMakeLists.txt` 的基本结构
- 理解 out-of-source build
- 理解构建目标、选项以及 debug/release 模式
- 理解如何添加一个新的源文件或库目标

## 为什么重要

这套教程本身就是一个 C++ 项目，所以构建系统也是学习路径的一部分。如果读者不理解 target、include path 和独立 build 目录，后面的章节就会更难编译、测试和扩展。

在这个项目里，构建系统不是可有可无的，因为每一章都应该能独立运行。如果你分不清哪个 target 负责哪个二进制、哪个目录保存生成物、哪个选项控制 backend，调试教程会变得很贵。

## 后面会用到什么

- 构建 `llama.cpp`
- 构建教程代码
- 添加测试和 benchmark target
- 管理可选的 CUDA 编译

## 这一页的核心内容

### 1. 项目和目标

`project()`、`add_executable()` 和 `add_library()` 定义了构建图。

实际写项目时，每一章都应该把自己的 targets 暴露清楚，这样读者才能只构建某一章，而不是在脑子里加载整个仓库。

同时，教程项目最好尽早统一 C++ 标准，避免不同章节各自带一套编译假设：

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```

这样读者能更明确地知道，这套教程依赖的是标准 C++，不是某个编译器的 GNU 扩展。

### 2. Include Path 和 Link Library

`target_include_directories()` 和 `target_link_libraries()` 让依赖只留在真正需要它的 target 上。

这种局部性对教程尤其重要，因为按章节演进时，依赖如果是显式的，而不是全局的，就更容易理解。

更常见的实践是把公开头文件放进专门的 `include/` 目录，而不是直接暴露整个源码目录。这样一来，`public header` 和 `internal header` 的边界会更清楚。

```cmake
target_include_directories(engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
```

如果某些教程示例暂时还没拆出 `include/` 目录，也可以先用当前源码目录过渡，但应该明确告诉读者这只是简化版写法，不是最终推荐形态。

### 3. 构建选项

用选项切换功能，比如 CPU-only 和 CUDA-enabled 构建。

构建选项本身也是教程的一部分：读者应该能看出来，哪些功能是基础路径，哪些是加速路径。

一个最小的选项示例就足够说明问题：

```cmake
option(ENABLE_CUDA "Enable CUDA support" OFF)
if(ENABLE_CUDA)
    target_compile_definitions(engine PRIVATE USE_CUDA)
endif()
```

这类写法的重点不是把功能拆得很细，而是让读者知道“构建期开关”怎么映射到“代码路径”。

### 4. Out-of-Source Build

把生成文件放到专门的 build 目录，避免源码和构建产物混在一起。

这样更容易干净地重跑某一章、对比不同 build 版本，也不容易把生成文件和手写代码搞混。

你可以把它理解成一个基本约束：

- `src/` 放手写代码
- `build/` 放自动生成物
- 目录边界清楚，重建和清理才不会互相污染

### 5. Debug 和 Release

理解为什么 debug build 更容易检查，而 release build 更适合做性能验证。

这个区分到后面很重要，因为正确性和性能不是用同一个 build profile 来验证的。

## 最小示例

```cmake
cmake_minimum_required(VERSION 3.20)
project(mini_engine LANGUAGES CXX)

add_library(engine core.cpp)
target_compile_features(engine PUBLIC cxx_std_17)
target_include_directories(engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(app main.cpp)
target_link_libraries(app PRIVATE engine)
```

这个例子展示了一个小项目的基本形状：

- 一个 library target
- 一个 executable target
- 明确的 include path
- 明确的链接关系

它背后的隐藏含义是：构建图本身也可以像文档一样工作。如果一章的 targets 很好懂，代码库就会更容易扩展。

## 怎么验证

- 能成功配置
- 能成功构建
- 能成功运行可执行文件
- 能确认新增一个源文件时，不需要重构整个项目布局
- 能用独立的 `build/` 目录重复构建
- 能只构建单个 target，而不是每次都重建整个仓库
- 能把新增源文件放进合适的目录，而不是把源码和产物混在一起

如果这些检查都通过，读者就应该能继续添加章节模块或测试 target，而不用先重写 build system。

常用的增量构建命令可以直接记住这两个：

```bash
cmake --build build --target engine
cmake --build build --target clean
```

前者只构建特定目标，后者清理后重新构建。对教程开发来说，这比每次全量重跑更实用。

## 常见问题

- 源文件和 build 文件混在同一个目录
- 使用全局 include path，而不是 target-local 的
- 链接库之前忘了先把符号对应起来
- 把 feature flag 写散在很多地方，而不是统一成一个选项
- 公开头文件和内部头文件没有分层
- 一开始就暴露整个源码目录，后面很难收窄

## 源文件组织

如果章节开始长出多个模块，建议尽早让目录结构和 CMake 结构对齐：

```text
src/
├── core/
│   ├── CMakeLists.txt
│   └── tensor.cpp
├── ops/
│   ├── CMakeLists.txt
│   └── matmul.cpp
└── main.cpp
```

```cmake
add_subdirectory(src/core)
add_subdirectory(src/ops)
```

这不是要求 `00.2` 立刻变复杂，而是告诉读者：一旦项目开始增长，模块化组织是正常演进，不是额外负担。

## 总结

这一章提供了理解项目构建方式所需的基础知识。读完后，读者应该能看懂教程代码是怎么被构建出来的，以及后续该把新模块或测试放到哪里。

如果你能解释 target graph、build 目录布局和主要 feature flag，那么这套构建系统就已经足够支撑后面的教程了。
