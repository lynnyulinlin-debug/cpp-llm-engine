# Chapter 03：MatMul CPU 构建入口
# 构建：生成 core、MatMul library、demo 和联合 CTest。
cmake_minimum_required(VERSION 3.20)
project(chapter03_matmul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CMAKE_CURRENT_BINARY_DIR}/core
)

add_library(chapter03_matmul
    src/matrix.cpp
    src/operator.cpp
    src/cpu_kernel.cpp
    src/cpu_backend.cpp
    src/matmul_op.cpp
)

target_include_directories(chapter03_matmul
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chapter03_matmul
    PUBLIC
        cpp_llm_engine_core
)

target_compile_features(chapter03_matmul PUBLIC cxx_std_17)

add_executable(chapter03_matmul_demo
    examples/main_matmul.cpp
)

target_link_libraries(chapter03_matmul_demo
    PRIVATE chapter03_matmul cpp_llm_engine_core
)

add_executable(chapter03_matmul_test
    tests/chapter03_matmul_test.cpp
    tests/operator_smoke_test.cpp
)

target_link_libraries(chapter03_matmul_test
    PRIVATE chapter03_matmul cpp_llm_engine_core
)

include(CTest)
enable_testing()

add_test(NAME chapter03_matmul_test COMMAND chapter03_matmul_test)
