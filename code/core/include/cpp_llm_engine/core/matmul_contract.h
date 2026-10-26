// Core：MatMul backend 请求契约
// 职责：固定 lhs[M,K] * rhs[K,N] = output[M,N] 的 row-major 语义。
// 契约：backend 校验 buffer 和 shape，并写入 output。
// 验证：Chapter 03 MatMul 和 core contract 测试。
#pragma once

#include "backend.h"
#include "status.h"

#include <cstddef>

namespace cpp_llm_engine::core {

// Row-major contract: lhs[M,K] * rhs[K,N] = output[M,N].
struct MatmulRequest final {
    const float* lhs = nullptr;
    const float* rhs = nullptr;
    float* output = nullptr;
    std::size_t rows = 0;
    std::size_t inner = 0;
    std::size_t cols = 0;
};

class MatmulBackend : public Backend {
public:
    ~MatmulBackend() override = default;
    virtual Status compute(const MatmulRequest& request) const = 0;
};

}  // namespace cpp_llm_engine::core
