// Core：Attention backend 请求契约
// 职责：固定 row-major Q/K/V 的 token、head 和 head_dim 布局。
// 契约：backend 读取 query/key/value，并将结果写入 output。
// 验证：code/core/tests/core_contract_test.cpp 及 Chapter 04 Attention 测试。
#pragma once

#include "backend.h"
#include "status.h"

#include <cstddef>

namespace cpp_llm_engine::core {

// Row-major Q/K/V layout: [tokens, heads * head_dim].
struct AttentionRequest final {
    const float* query = nullptr;
    const float* key = nullptr;
    const float* value = nullptr;
    float* output = nullptr;
    std::size_t query_tokens = 0;
    std::size_t key_tokens = 0;
    std::size_t heads = 0;
    std::size_t head_dim = 0;
    std::size_t query_position_offset = 0;
    bool causal = true;
    float scale = 0.0f;
};

class AttentionBackend : public Backend {
public:
    ~AttentionBackend() override = default;
    virtual Status compute(const AttentionRequest& request) const = 0;
};

}  // namespace cpp_llm_engine::core
