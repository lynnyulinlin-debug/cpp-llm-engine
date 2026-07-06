cmake_minimum_required(VERSION 3.20)
project(chapter01_tensor_loader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chapter01_tensor_loader
    src/tensor.cpp
    src/model_loader.cpp
)

target_include_directories(chapter01_tensor_loader
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(chapter01_tensor_loader_test
    tests/chapter01_tensor_loader_test.cpp
)

target_link_libraries(chapter01_tensor_loader_test
    PRIVATE chapter01_tensor_loader
)

enable_testing()
add_test(
    NAME chapter01_tensor_loader_test
    COMMAND chapter01_tensor_loader_test
)
