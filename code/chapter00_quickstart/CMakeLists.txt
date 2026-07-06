cmake_minimum_required(VERSION 3.20)

project(chapter00_quickstart LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(chapter00_quickstart
    main_quick.cpp
)

target_compile_features(chapter00_quickstart PRIVATE cxx_std_17)
