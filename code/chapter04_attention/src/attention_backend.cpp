// Chapter 04：CPU Attention backend 实现
// 职责：提供 CPU backend 身份，并将 AttentionRequest 分派给 reference kernel。
// 契约：不重复实现 softmax 或 mask，错误由 kernel 返回 Status。
// 验证：chapter04_attention/scripts/run.sh。
#include "chapter04_attention/attention_backend.h"
#include "chapter04_attention/attention_kernel.h"

namespace chapter04_attention {

cpp_llm_engine::core::BackendKind CpuReferenceAttentionBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view CpuReferenceAttentionBackend::name() const {
    return "cpu-reference-attention";
}

cpp_llm_engine::core::Status CpuReferenceAttentionBackend::compute(
    const cpp_llm_engine::core::AttentionRequest& request) const {
    return cpu_reference_attention_kernel(request);
}

const CpuReferenceAttentionBackend& cpu_reference_attention_backend() {
    static const CpuReferenceAttentionBackend backend;
    return backend;
}

}  // namespace chapter04_attention
