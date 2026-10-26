cmake_minimum_required(VERSION 3.20)
project(chapter06_quantization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT TARGET cpp_llm_engine_core)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../core
        ${CMAKE_CURRENT_BINARY_DIR}/core
    )
endif()

add_library(chapter06_quantization
    src/dequantize_op.cpp
    src/linear_layer.cpp
    src/quantized_linear.cpp
    src/q4k_dequantize.cpp
    src/q4k_kernel.cpp
    src/q4k_linear.cpp
    src/q6k_dequantize.cpp
    src/q6k_kernel.cpp
    src/q6k_linear.cpp
    src/q8_0_dequantize.cpp
    src/q8_0_linear.cpp
    src/quantized_linear_policy.cpp
    src/tensor.cpp
)

target_include_directories(chapter06_quantization
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chapter06_quantization
    PUBLIC
        cpp_llm_engine_core
)

target_compile_features(chapter06_quantization PUBLIC cxx_std_17)

add_executable(chapter06_quantization_demo
    examples/main_quantization.cpp
)

add_executable(q4k_benchmark
    examples/q4k_benchmark.cpp
)

target_link_libraries(q4k_benchmark
    PRIVATE chapter06_quantization
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    add_library(chapter06_quantization_avx2
        src/q4k_linear_avx2.cpp
        src/q6k_linear_avx2.cpp
    )
    target_include_directories(chapter06_quantization_avx2
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_options(chapter06_quantization_avx2 PRIVATE -mavx2 -mfma)
    target_link_libraries(chapter06_quantization_avx2 PUBLIC chapter06_quantization)
    target_compile_definitions(q4k_benchmark PRIVATE CHAPTER06_HAS_AVX2=1)
    target_link_libraries(q4k_benchmark PRIVATE chapter06_quantization_avx2)
endif()

target_link_libraries(chapter06_quantization_demo
    PRIVATE chapter06_quantization
)

add_executable(chapter06_quantization_test
    tests/chapter06_quantization_test.cpp
)

target_link_libraries(chapter06_quantization_test
    PRIVATE chapter06_quantization
)

if(TARGET chapter06_quantization_avx2)
    target_compile_definitions(chapter06_quantization_test PRIVATE CHAPTER06_HAS_AVX2=1)
    target_link_libraries(chapter06_quantization_test PRIVATE chapter06_quantization_avx2)
endif()

include(CTest)
enable_testing()

add_test(NAME chapter06_quantization_test COMMAND chapter06_quantization_test)
