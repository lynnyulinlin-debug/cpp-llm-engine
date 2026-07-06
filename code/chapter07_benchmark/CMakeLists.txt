cmake_minimum_required(VERSION 3.20)
project(chapter07_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chapter07_benchmark
    src/benchmark.cpp
    src/test_helpers.cpp
)

target_include_directories(chapter07_benchmark
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chapter07_benchmark PUBLIC cxx_std_17)

add_executable(chapter07_benchmark_demo
    examples/main_benchmark.cpp
)

target_link_libraries(chapter07_benchmark_demo
    PRIVATE chapter07_benchmark
)

add_executable(chapter07_benchmark_test
    tests/chapter07_benchmark_test.cpp
)

target_link_libraries(chapter07_benchmark_test
    PRIVATE chapter07_benchmark
)

include(CTest)
enable_testing()

add_test(NAME chapter07_benchmark_test COMMAND chapter07_benchmark_test)
