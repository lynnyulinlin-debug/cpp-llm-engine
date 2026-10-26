// Core：Elementwise 请求契约
// 职责：固定等长 Float32 Vector Add 的输入、输出和元素数量。
// 契约：指针属于所选 backend 的设备地址空间，backend 负责参数校验。
// 验证：CUDA elementwise CTest 与 CPU golden 逐元素对照。
#pragma once

#include <cstddef>

namespace cpp_llm_engine::core {

struct VectorAddRequest final {
    const float* lhs = nullptr;
    const float* rhs = nullptr;
    float* output = nullptr;
    std::size_t elements = 0;
};

}  // namespace cpp_llm_engine::core
