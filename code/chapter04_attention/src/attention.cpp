// Chapter 04：MultiHeadAttention 实现
// 职责：执行 Q/K/V projection、Attention backend 调用和 output projection。
// 契约：输入 layout 为 [tokens, model_dim]，backend 看到连续 row-major buffer。
// 验证：chapter04_attention_test.cpp。
#include "chapter04_attention/attention.h"
#include "chapter04_attention/attention_backend.h"

#include <cmath>

namespace chapter04_attention {

MultiHeadAttention::MultiHeadAttention(std::size_t model_dim, std::size_t num_heads)
    : model_dim_(model_dim),
      num_heads_(num_heads),
      q_proj_(model_dim, model_dim),
      k_proj_(model_dim, model_dim),
      v_proj_(model_dim, model_dim),
      out_proj_(model_dim, model_dim),
      backend_(&cpu_reference_attention_backend()) {}

MultiHeadAttention::MultiHeadAttention(
    std::size_t model_dim,
    std::size_t num_heads,
    const cpp_llm_engine::core::AttentionBackend& backend)
    : model_dim_(model_dim),
      num_heads_(num_heads),
      q_proj_(model_dim, model_dim),
      k_proj_(model_dim, model_dim),
      v_proj_(model_dim, model_dim),
      out_proj_(model_dim, model_dim),
      backend_(&backend) {}

Tensor MultiHeadAttention::forward(const Tensor& input) const {
    if (!ready() || !input.ready() || input.cols != model_dim_) {
        return {};
    }

    const std::size_t head_dim = model_dim_ / num_heads_;
    Tensor q = q_proj_.forward(input);
    Tensor k = k_proj_.forward(input);
    Tensor v = v_proj_.forward(input);

    Tensor combined(input.rows, model_dim_);
    const cpp_llm_engine::core::AttentionRequest request{
        q.data.data(), k.data.data(), v.data.data(), combined.data.data(),
        input.rows, input.rows, num_heads_, head_dim, 0, true, 0.0f};
    const auto status = backend_->compute(request);
    if (!status.is_ok()) return {};
    return out_proj_.forward(combined);
}

bool MultiHeadAttention::ready() const {
    return model_dim_ > 0 && num_heads_ > 0 && model_dim_ % num_heads_ == 0 &&
           q_proj_.ready() && k_proj_.ready() && v_proj_.ready() && out_proj_.ready();
}

std::size_t MultiHeadAttention::model_dim() const {
    return model_dim_;
}

std::size_t MultiHeadAttention::num_heads() const {
    return num_heads_;
}

cpp_llm_engine::core::BackendKind MultiHeadAttention::backend_kind() const {
    return backend_->kind();
}

}  // namespace chapter04_attention
