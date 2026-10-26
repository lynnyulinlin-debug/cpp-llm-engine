// CUDA Backend：声明可解释的 causal GQA Attention backend。
// 职责：依次计算 score、稳定 softmax 和 value aggregation，并使用外部 workspace。
// 验证：prefill/decode CPU golden、GQA head 映射、错误参数和 event 计时。
#pragma once

#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/core/gqa_attention_contract.h"
#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>
#include <string_view>

namespace cpp_llm_engine::cuda {

class CudaStream;

class CudaGqaAttentionBackend final : public core::Backend {
public:
    core::BackendKind kind() const override;
    std::string_view name() const override;

    CudaStatus compute(
        const core::GqaAttentionRequest& request,
        float* score_workspace,
        std::size_t workspace_elements,
        const CudaStream& stream) const;
};

const CudaGqaAttentionBackend& cuda_gqa_attention_backend();

}  // namespace cpp_llm_engine::cuda
