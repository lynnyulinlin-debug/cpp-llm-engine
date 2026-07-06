cmake_minimum_required(VERSION 3.20)
project(chapter06_quantization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chapter06_quantization
    src/dequantize_op.cpp
    src/linear_layer.cpp
    src/quantized_linear.cpp
    src/tensor.cpp
)

target_include_directories(chapter06_quantization
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chapter06_quantization PUBLIC cxx_std_17)

add_executable(chapter06_quantization_demo
    examples/main_quantization.cpp
)

target_link_libraries(chapter06_quantization_demo
    PRIVATE chapter06_quantization
)

add_executable(chapter06_quantization_test
    tests/chapter06_quantization_test.cpp
)

target_link_libraries(chapter06_quantization_test
    PRIVATE chapter06_quantization
)

include(CTest)
enable_testing()

add_test(NAME chapter06_quantization_test COMMAND chapter06_quantization_test)
