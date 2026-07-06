cmake_minimum_required(VERSION 3.20)
project(chapter04_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chapter04_attention
    src/attention.cpp
    src/decoder_layer.cpp
    src/feed_forward.cpp
    src/layer_norm.cpp
    src/linear_layer.cpp
    src/tensor.cpp
)

target_include_directories(chapter04_attention
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
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
