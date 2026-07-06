# F09 Minimal C++ / CUDA Project

## Goal

Show what the smallest usable C++ project should look like first, then show how to extend it to CUDA without rewriting the layout.

## What you should be able to do after reading

- Understand the directory layout of a minimal C++ project
- Understand how `main.cpp`, headers, source files, and `CMakeLists.txt` work together
- Understand how to add a library target before adding CUDA
- Understand how to switch from CPU-only to CUDA-enabled builds

## Why this matters

Many beginners get stuck before writing real code because the project layout feels too large. This page takes the opposite approach: start with one file, one target, one executable, and only grow when needed.

## What comes next

- The tutorial’s `code/chapterXX_*` projects
- CPU and CUDA build paths
- library targets and test targets
- adding source files without breaking the build

## Relation to `F02`

`F02` introduces `project()`, `add_executable()`, `add_library()`, and `target_link_libraries()`.  
`F09` combines those ideas into a small, runnable project.

## Core ideas

### 1. Start with one executable

A minimal useful project can be just a `main.cpp` and one executable target. That gives you a first successful build before you add more structure.

### 2. Split into library and app

Once the executable works, move reusable logic into a library target. That is the normal path from demo code to chapter modules.

### 3. Add headers only when needed

Keep declarations in headers and implementations in source files. Let the interface boundary stay small.

### 4. Treat CUDA as a target extension

CPU first, then a CUDA source file and a build option. CUDA should be an extension of the same project, not a separate puzzle.

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

## Minimal example

### Layout

```text
mini_project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── core.cpp
│   ├── core.h
│   └── kernel.cu
```

### CPU-only version

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

### CUDA extension

```cmake
option(ENABLE_CUDA "Enable CUDA support" OFF)

if(ENABLE_CUDA)
    enable_language(CUDA)
    add_library(cuda_kernels src/kernel.cu)
    target_link_libraries(app PRIVATE cuda_kernels)
endif()
```

## How to validate

- Build the CPU-only project successfully
- Run the executable successfully
- Add one source file without changing the whole layout
- Enable CUDA while keeping the CPU path intact
- Compare CPU-only and CUDA-enabled build logs

## Summary

If you can build the CPU version cleanly, add one reusable library, and then turn CUDA into an option rather than a rewrite, the project structure is good enough for the rest of the tutorial.
