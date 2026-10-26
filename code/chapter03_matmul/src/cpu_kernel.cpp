// Chapter 03：CPU MatMul reference kernel 实现
// 职责：执行 row-major 三重循环并校验 buffer、shape 和尺寸溢出。
// 契约：保持公共 MatmulRequest 语义，作为 SIMD/cuBLAS 的数值基线。
// 验证：chapter03_matmul_test.cpp。
#include "chapter03_matmul/cpu_kernel.h"

#include <limits>

namespace chapter03_matmul {

namespace {

bool product_fits_host(std::size_t lhs, std::size_t rhs) {
    return rhs == 0 || lhs <= std::numeric_limits<std::size_t>::max() / rhs;
}

}  // namespace

cpp_llm_engine::core::Status cpu_reference_matmul_kernel(
    const cpp_llm_engine::core::MatmulRequest& request) {
    if (request.lhs == nullptr || request.rhs == nullptr || request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("matmul buffers must not be null");
    }
    if (request.rows == 0 || request.inner == 0 || request.cols == 0) {
        return cpp_llm_engine::core::Status::error("matmul dimensions must be non-zero");
    }
    if (!product_fits_host(request.rows, request.inner) ||
        !product_fits_host(request.inner, request.cols) ||
        !product_fits_host(request.rows, request.cols)) {
        return cpp_llm_engine::core::Status::error("matmul buffer size overflows host size");
    }

    for (std::size_t row = 0; row < request.rows; ++row) {
        for (std::size_t col = 0; col < request.cols; ++col) {
            float value = 0.0f;
            for (std::size_t inner = 0; inner < request.inner; ++inner) {
                value += request.lhs[row * request.inner + inner] *
                         request.rhs[inner * request.cols + col];
            }
            request.output[row * request.cols + col] = value;
        }
    }
    return cpp_llm_engine::core::Status::ok();
}

}  // namespace chapter03_matmul
