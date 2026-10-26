// Core：Linear backend 请求契约
// 职责：固定 row-major weights[rows, cols] * input[cols] 的输入输出布局。
// 契约：weight_format 标识 Float32、Q4K、Q6K 或 Q8_0，backend 写入 output。
// 验证：Chapter 06 quantized backend 测试。
#pragma once

#include "backend.h"
#include "status.h"

#include <cstddef>
#include <cstdint>

namespace cpp_llm_engine::core {

enum class WeightFormat : std::uint8_t {
    Unknown = 0,
    Float32,
    Q4K,
    Q6K,
    Q8_0,
};

// Row-major contract: weights[rows, cols] * input[cols] = output[rows].
struct LinearRequest final {
    const void* weights = nullptr;
    std::size_t weight_bytes = 0;
    const float* input = nullptr;
    std::size_t input_size = 0;
    float* output = nullptr;
    std::size_t output_size = 0;
    std::size_t rows = 0;
    std::size_t cols = 0;
    WeightFormat weight_format = WeightFormat::Unknown;
};

class LinearBackend : public Backend {
public:
    ~LinearBackend() override = default;
    virtual Status compute(const LinearRequest& request) const = 0;
};

}  // namespace cpp_llm_engine::core
