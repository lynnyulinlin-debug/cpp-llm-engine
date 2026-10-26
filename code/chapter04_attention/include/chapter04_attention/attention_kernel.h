// Chapter 04：CPU Attention kernel 接口
// 职责：声明 row-major Q/K/V、causal mask 和 stable softmax 的 reference kernel。
// 契约：调用方拥有 buffer，非法尺寸或不可见位置返回 Status 错误。
// 验证：chapter04_attention_test.cpp。
#pragma once

#include "cpp_llm_engine/core/attention_contract.h"

namespace chapter04_attention {

// Row-major CPU reference attention with optional causal masking.
cpp_llm_engine::core::Status cpu_reference_attention_kernel(
    const cpp_llm_engine::core::AttentionRequest& request);

}  // namespace chapter04_attention
