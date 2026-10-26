cmake_minimum_required(VERSION 3.16)
project(fq17_minimal_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(foundation_engine src/engine.cpp)
target_compile_options(foundation_engine PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(foundation_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(engine_demo app/main.cpp)
target_link_libraries(engine_demo PRIVATE foundation_engine)

add_executable(engine_test tests/engine_test.cpp)
target_link_libraries(engine_test PRIVATE foundation_engine)

enable_testing()
add_test(NAME engine_test COMMAND engine_test)
