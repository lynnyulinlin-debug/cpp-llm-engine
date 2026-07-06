#pragma once

#include "chapter04_attention/attention.h"
#include "chapter04_attention/feed_forward.h"
#include "chapter04_attention/layer_norm.h"

#include <cstddef>

namespace chapter04_attention {

class DecoderLayer final {
public:
    DecoderLayer(std::size_t model_dim = 0, std::size_t num_heads = 0, std::size_t ffn_dim = 0, std::size_t layer_index = 0);

    Tensor forward(const Tensor& input) const;
    bool ready() const;

    std::size_t layer_index() const;

private:
    std::size_t layer_index_ = 0;
    LayerNorm pre_attn_norm_;
    MultiHeadAttention attention_;
    LayerNorm pre_ffn_norm_;
    FeedForward feed_forward_;
};

}  // namespace chapter04_attention
