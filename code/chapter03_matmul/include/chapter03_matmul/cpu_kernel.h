// Chapter 03：CPU MatMul kernel 接口
// 职责：声明 row-major MxK * KxN 的可读 CPU reference kernel。
// 契约：调用方拥有输入输出 buffer，kernel 返回 Status 而不分配矩阵对象。
// 验证：chapter03_matmul_test.cpp。
#pragma once

#include "cpp_llm_engine/core/matmul_contract.h"

namespace chapter03_matmul {

// Row-major MxK times KxN reference kernel. The caller owns all buffers.
cpp_llm_engine::core::Status cpu_reference_matmul_kernel(
    const cpp_llm_engine::core::MatmulRequest& request);

}  // namespace chapter03_matmul
