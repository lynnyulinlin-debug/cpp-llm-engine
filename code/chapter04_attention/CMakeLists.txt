# Chapter 04：Attention/Decoder CPU 构建入口
# 构建：生成 core、Attention library、Decoder demo 和联合 CTest。
cmake_minimum_required(VERSION 3.20)
project(chapter04_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CMAKE_CURRENT_BINARY_DIR}/core
)

add_library(chapter04_attention
    src/attention.cpp
    src/attention_backend.cpp
    src/attention_kernel.cpp
    src/decoder_layer.cpp
    src/feed_forward.cpp
    src/layer_norm.cpp
    src/linear_layer.cpp
    src/kv_cache.cpp
    src/tensor.cpp
)

target_include_directories(chapter04_attention
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chapter04_attention
    PUBLIC
        cpp_llm_engine_core
)

target_compile_features(chapter04_attention PUBLIC cxx_std_17)

add_executable(chapter04_attention_demo
    examples/main_attention.cpp
)

target_link_libraries(chapter04_attention_demo
    PRIVATE chapter04_attention
)

add_executable(chapter04_attention_test
    tests/chapter04_attention_test.cpp
)

target_link_libraries(chapter04_attention_test
    PRIVATE chapter04_attention
)

include(CTest)
enable_testing()

add_test(NAME chapter04_attention_test COMMAND chapter04_attention_test)
