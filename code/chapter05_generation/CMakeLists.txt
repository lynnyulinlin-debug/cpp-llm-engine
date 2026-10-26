# Chapter 05：构建 Generation Runtime、示例和专项测试。
# 目标：让 Request、KV Cache、Scheduler 和 Runtime 保持可独立验证。
cmake_minimum_required(VERSION 3.20)
project(chapter05_generation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chapter05_generation
    src/request.cpp
    src/sampler.cpp
    src/kv_cache.cpp
    src/runtime.cpp
    src/scheduler.cpp
    src/embedding.cpp
    src/position_encoding.cpp
    src/decoder_layer.cpp
    src/serving_preview.cpp
    src/transformer.cpp
)

target_include_directories(chapter05_generation
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chapter05_generation PUBLIC cxx_std_17)

add_executable(chapter05_generation_demo
    examples/main_generation.cpp
)

target_link_libraries(chapter05_generation_demo
    PRIVATE chapter05_generation
)

add_executable(chapter05_generation_test
    tests/chapter05_generation_test.cpp
)

target_link_libraries(chapter05_generation_test
    PRIVATE chapter05_generation
)

add_executable(chapter05_runtime_smoke_test
    tests/runtime_smoke_test.cpp
)

target_link_libraries(chapter05_runtime_smoke_test
    PRIVATE chapter05_generation
)

add_executable(chapter05_sampler_test
    tests/sampler_test.cpp
)

target_link_libraries(chapter05_sampler_test
    PRIVATE chapter05_generation
)

add_executable(chapter05_kv_cache_test
    tests/kv_cache_test.cpp
)

target_link_libraries(chapter05_kv_cache_test
    PRIVATE chapter05_generation
)

add_executable(chapter05_scheduler_test
    tests/chapter05_scheduler_test.cpp
)

target_link_libraries(chapter05_scheduler_test
    PRIVATE chapter05_generation
)

add_executable(chapter05_serving_preview_test
    tests/serving_preview_test.cpp
)

target_link_libraries(chapter05_serving_preview_test
    PRIVATE chapter05_generation
)

add_executable(chapter05_streaming_demo
    examples/main_streaming.cpp
)

target_link_libraries(chapter05_streaming_demo
    PRIVATE chapter05_generation
)

include(CTest)
enable_testing()

add_test(NAME chapter05_generation_test COMMAND chapter05_generation_test)
add_test(NAME chapter05_runtime_smoke_test COMMAND chapter05_runtime_smoke_test)
add_test(NAME chapter05_sampler_test COMMAND chapter05_sampler_test)
add_test(NAME chapter05_kv_cache_test COMMAND chapter05_kv_cache_test)
add_test(NAME chapter05_scheduler_test COMMAND chapter05_scheduler_test)
add_test(NAME chapter05_serving_preview_test COMMAND chapter05_serving_preview_test)
