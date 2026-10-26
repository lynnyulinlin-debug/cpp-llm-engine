// Chapter 04：Attention backend 接口
// 职责：声明 CPU Attention backend 的设备身份和 AttentionRequest 执行入口。
// 契约：backend 保持 Q/K/V row-major 布局和 causal/scale 语义。
// 验证：chapter04_attention_test.cpp 和 core CTest。
#pragma once

#include "cpp_llm_engine/core/attention_contract.h"

namespace chapter04_attention {

class CpuReferenceAttentionBackend final : public cpp_llm_engine::core::AttentionBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::AttentionRequest& request) const override;
};

const CpuReferenceAttentionBackend& cpu_reference_attention_backend();

}  // namespace chapter04_attention
