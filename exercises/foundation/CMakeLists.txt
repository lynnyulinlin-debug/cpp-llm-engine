cmake_minimum_required(VERSION 3.16)
project(foundation_exercises LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(ENABLE_ASAN AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

enable_testing()

function(add_foundation_exercise name source)
  add_executable(${name} ${source})
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

add_foundation_exercise(fq01_move_buffer fq01_move_buffer.cpp)
add_foundation_exercise(fq02_tensor_shape fq02_tensor_shape.cpp)
add_foundation_exercise(fq03_tensor_view fq03_tensor_view.cpp)
add_foundation_exercise(fq04_cmake_ctest fq04_cmake_ctest.cpp)
add_foundation_exercise(fq05_debug_safe_buffer fq05_debug_safe_buffer.cpp)
add_foundation_exercise(fq06_cpu_matmul fq06_cpu_matmul.cpp)
add_foundation_exercise(fq07_token_queue fq07_token_queue.cpp)
add_foundation_exercise(fq08_mini_tensor fq08_mini_tensor.cpp)
