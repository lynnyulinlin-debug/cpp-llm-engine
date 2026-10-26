// Core：Grouped-Query Attention backend 请求契约。
// 职责：固定 Q 与 K/V 不同 head 数、causal position 和缩放参数。
// 契约：Q/output 为 [Tq,Hq,Dh]，K/V 为 [Tk,Hkv,Dh]，Hq 必须能被 Hkv 整除。
// 验证：Core contract test 与 CUDA GQA Attention CPU golden 对照。
#pragma once

#include "status.h"

#include <cmath>
#include <cstddef>
#include <limits>

namespace cpp_llm_engine::core {

struct GqaAttentionRequest final {
    const float* query = nullptr;
    const float* key = nullptr;
    const float* value = nullptr;
    float* output = nullptr;
    std::size_t query_tokens = 0;
    std::size_t key_tokens = 0;
    std::size_t query_head_count = 0;
    std::size_t kv_head_count = 0;
    std::size_t head_dimension = 0;
    std::size_t query_position_offset = 0;
    bool causal = true;
    float scale = 0.0F;

    Status validate() const {
        if (query == nullptr || key == nullptr || value == nullptr || output == nullptr) {
            return Status::error("GQA buffers must not be null");
        }
        if (query_tokens == 0 || key_tokens == 0 || query_head_count == 0 ||
            kv_head_count == 0 || head_dimension == 0) {
            return Status::error("GQA dimensions must be non-zero");
        }
        if (query_head_count % kv_head_count != 0) {
            return Status::error("GQA query heads must be divisible by KV heads");
        }
        if (!std::isfinite(scale) || scale < 0.0F) {
            return Status::error("GQA scale must be finite and non-negative");
        }
        if (query_position_offset >
            std::numeric_limits<std::size_t>::max() - (query_tokens - 1)) {
            return Status::error("GQA query position overflows");
        }
        if (query_head_count >
                std::numeric_limits<std::size_t>::max() / head_dimension ||
            kv_head_count >
                std::numeric_limits<std::size_t>::max() / head_dimension) {
            return Status::error("GQA head width overflows");
        }
        const std::size_t query_width = query_head_count * head_dimension;
        const std::size_t kv_width = kv_head_count * head_dimension;
        if (query_tokens > std::numeric_limits<std::size_t>::max() / query_width ||
            key_tokens > std::numeric_limits<std::size_t>::max() / kv_width ||
            query_tokens > std::numeric_limits<std::size_t>::max() / query_head_count ||
            query_tokens * query_head_count >
                std::numeric_limits<std::size_t>::max() / key_tokens) {
            return Status::error("GQA element count overflows");
        }
        return Status::ok();
    }

    std::size_t score_count() const {
        return query_tokens * query_head_count * key_tokens;
    }
};

}  // namespace cpp_llm_engine::core
