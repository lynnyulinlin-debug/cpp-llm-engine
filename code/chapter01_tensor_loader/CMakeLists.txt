cmake_minimum_required(VERSION 3.20)
project(chapter01_tensor_loader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CHAPTER01_ENABLE_LLAMA_CPP "Build the real llama.cpp tokenizer adapter" OFF)
set(LLAMA_CPP_ROOT "" CACHE PATH "llama.cpp checkout containing include/ and build/bin/")
set(LLAMA_CPP_BUILD_DIR "" CACHE PATH "Optional llama.cpp build directory containing bin/")

add_library(chapter01_tensor_loader
    src/tensor.cpp
    src/model_loader.cpp
    src/model_config.cpp
    src/model_metadata.cpp
    src/model_schema.cpp
    src/model_metadata_adapter.cpp
    src/gguf_reader.cpp
    src/gguf_tensor_loader.cpp
    src/model_builder.cpp
    src/tokenizer_metadata.cpp
    src/tokenizer.cpp
    src/real_layer_weights.cpp
)

target_include_directories(chapter01_tensor_loader
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(chapter01_tensor_loader_test
    tests/chapter01_tensor_loader_test.cpp
)

add_executable(gguf_inspect
    examples/gguf_inspect.cpp
)

add_executable(gguf_first_layer_inspect
    examples/gguf_first_layer_inspect.cpp
)

target_link_libraries(gguf_inspect
    PRIVATE chapter01_tensor_loader
)

target_link_libraries(gguf_first_layer_inspect
    PRIVATE chapter01_tensor_loader
)

target_link_libraries(chapter01_tensor_loader_test
    PRIVATE chapter01_tensor_loader
)

# Keep Chapter 01 standalone, but allow an explicit cross-chapter integration
# build that exercises the real GGUF payload through Chapter 06's Q4_K Linear.
set(CHAPTER06_SOURCE_DIR "" CACHE PATH "Chapter 06 source directory for integration tests")
if(CHAPTER06_SOURCE_DIR)
    if(NOT EXISTS "${CHAPTER06_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "CHAPTER06_SOURCE_DIR must point to the Chapter 06 source directory")
    endif()
    add_subdirectory("${CHAPTER06_SOURCE_DIR}" "chapter06_quantization")
    add_library(chapter01_real_decoder
        src/real_llama_cpu_decoder.cpp
    )
    target_include_directories(chapter01_real_decoder
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(chapter01_real_decoder
        PUBLIC
            chapter01_tensor_loader
            chapter06_quantization
    )
    if(TARGET chapter06_quantization_avx2)
        target_compile_definitions(chapter01_real_decoder
            PRIVATE
                CHAPTER01_HAS_Q4K_AVX2=1
                CHAPTER01_HAS_Q6K_AVX2=1
        )
        target_link_libraries(chapter01_real_decoder
            PUBLIC chapter06_quantization_avx2
        )
    endif()
    add_executable(gguf_q4k_linear_smoke
        examples/gguf_q4k_linear_smoke.cpp
    )
    target_link_libraries(gguf_q4k_linear_smoke
        PRIVATE
            chapter01_tensor_loader
            chapter06_quantization
    )
    add_executable(gguf_q4k_benchmark
        examples/gguf_q4k_benchmark.cpp
    )
    target_link_libraries(gguf_q4k_benchmark
        PRIVATE
            chapter01_tensor_loader
            chapter06_quantization
    )
endif()

enable_testing()
add_test(
    NAME chapter01_tensor_loader_test
    COMMAND chapter01_tensor_loader_test
)

if(CHAPTER01_ENABLE_LLAMA_CPP)
    if(NOT EXISTS "${LLAMA_CPP_ROOT}/include/llama.h")
        message(FATAL_ERROR "LLAMA_CPP_ROOT must point to a llama.cpp checkout")
    endif()
    if(NOT LLAMA_CPP_BUILD_DIR)
        set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_ROOT}/build")
    endif()
    if(NOT EXISTS "${LLAMA_CPP_BUILD_DIR}/bin")
        message(FATAL_ERROR "LLAMA_CPP_BUILD_DIR must contain bin/")
    endif()
    find_library(LLAMA_LIBRARY NAMES llama PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_LIBRARY NAMES ggml PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_BASE_LIBRARY NAMES ggml-base PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_CPU_LIBRARY NAMES ggml-cpu PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)

    add_library(chapter01_real_tokenizer
        src/real_tokenizer.cpp
    )
    target_include_directories(chapter01_real_tokenizer
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${LLAMA_CPP_ROOT}/include
            ${LLAMA_CPP_ROOT}/ggml/include
    )
    target_link_libraries(chapter01_real_tokenizer
        PUBLIC
            chapter01_tensor_loader
            ${LLAMA_LIBRARY}
            ${GGML_LIBRARY}
            ${GGML_CPU_LIBRARY}
            ${GGML_BASE_LIBRARY}
            ${CMAKE_DL_LIBS}
    )
    set_target_properties(chapter01_real_tokenizer PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )

    add_executable(llama_tokenizer_compare
        examples/llama_tokenizer_compare.cpp
    )
    target_link_libraries(llama_tokenizer_compare PRIVATE chapter01_real_tokenizer)
    set_target_properties(llama_tokenizer_compare PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )

    if(CHAPTER06_SOURCE_DIR)
    add_executable(gguf_q6k_compare
        examples/gguf_q6k_compare.cpp
    )
        target_include_directories(gguf_q6k_compare
            PRIVATE
                ${LLAMA_CPP_ROOT}/ggml/src
                ${LLAMA_CPP_ROOT}/ggml/include
        )
        target_link_libraries(gguf_q6k_compare
            PRIVATE
                chapter01_tensor_loader
                chapter06_quantization
                ${GGML_LIBRARY}
                ${GGML_CPU_LIBRARY}
                ${GGML_BASE_LIBRARY}
        )
        set_target_properties(gguf_q6k_compare PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
        )
    endif()
endif()

if(CHAPTER06_SOURCE_DIR)
    add_executable(gguf_first_layer_forward
        examples/gguf_first_layer_forward.cpp
    )
    target_link_libraries(gguf_first_layer_forward
        PRIVATE
            chapter01_tensor_loader
            chapter06_quantization
    )
    add_executable(gguf_first_layer_multitoken
        examples/gguf_first_layer_multitoken.cpp
    )
    target_link_libraries(gguf_first_layer_multitoken
        PRIVATE
            chapter01_tensor_loader
            chapter06_quantization
    )
    add_executable(gguf_decoder_multilayer
        examples/gguf_decoder_multilayer.cpp
    )
    target_link_libraries(gguf_decoder_multilayer
        PRIVATE
            chapter01_real_decoder
    )
    if(CHAPTER01_ENABLE_LLAMA_CPP)
        target_compile_definitions(gguf_decoder_multilayer
            PRIVATE CHAPTER01_COMPARE_LLAMA_CPP=1
        )
        target_include_directories(gguf_decoder_multilayer
            PRIVATE
                ${LLAMA_CPP_ROOT}/include
                ${LLAMA_CPP_ROOT}/ggml/include
        )
        target_link_libraries(gguf_decoder_multilayer
            PRIVATE
                ${LLAMA_LIBRARY}
                ${GGML_LIBRARY}
                ${GGML_CPU_LIBRARY}
                ${GGML_BASE_LIBRARY}
                ${CMAKE_DL_LIBS}
        )
        set_target_properties(gguf_decoder_multilayer PROPERTIES
            BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
        )
    endif()
endif()
