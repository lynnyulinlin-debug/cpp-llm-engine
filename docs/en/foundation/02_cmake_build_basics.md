# F02 CMake and Build Basics

## Goal

Understand how the tutorial project is configured and built.

## What you should be able to do after reading

- Understand the basic structure of a `CMakeLists.txt`
- Understand out-of-source builds
- Understand build targets, options, and debug/release modes
- Understand how to add a new source file or library target

## Why this matters

This tutorial is itself a C++ project, so the build system is part of the learning path. If you do not understand targets, include paths, and separate build directories, the later chapters become harder to compile, test, and extend.

## What comes next

- Building `llama.cpp`
- Building the tutorial code
- Adding tests and benchmark targets
- Managing optional CUDA compilation

## Core ideas

### 1. Project and target definitions

`project()`, `add_executable()`, and `add_library()` define the build graph.

Each chapter should expose its targets clearly so readers can build only what they need.

### 2. Target-local include paths and libraries

`target_include_directories()` and `target_link_libraries()` keep dependencies attached to the targets that actually need them.

When possible, public headers should live under a dedicated `include/` directory. That keeps public and internal headers clearly separated.

### 3. Build options

Use options to switch features such as CPU-only and CUDA-enabled builds.

```cmake
option(ENABLE_CUDA "Enable CUDA support" OFF)
if(ENABLE_CUDA)
    target_compile_definitions(engine PRIVATE USE_CUDA)
endif()
```

### 4. Out-of-source build

Keep generated files in a separate build directory. That keeps source files and build products from mixing.

### 5. Debug and release

Debug builds are easier to inspect. Release builds are better for performance checks. Later chapters rely on that difference.

## Minimal example

```cmake
cmake_minimum_required(VERSION 3.20)
project(mini_engine LANGUAGES CXX)

add_library(engine core.cpp)
target_compile_features(engine PUBLIC cxx_std_17)
target_include_directories(engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(app main.cpp)
target_link_libraries(app PRIVATE engine)
```

This shows the smallest useful shape:

- one library target
- one executable target
- explicit include path
- explicit link relation

## How to validate

- Configure successfully
- Build successfully
- Run the executable successfully
- Add one new source file without rewriting the project layout
- Rebuild from a clean `build/` directory
- Build only a single target

## Summary

This page gives the build-system base layer for the tutorial. If you can explain target graphs, build directory layout, and the main feature flags, the build system is good enough for the chapters that follow.
