#include "chapter04_attention/decoder_layer.h"

namespace chapter04_attention {

DecoderLayer::DecoderLayer(std::size_t model_dim, std::size_t num_heads, std::size_t ffn_dim, std::size_t layer_index)
    : layer_index_(layer_index),
      pre_attn_norm_(model_dim),
      attention_(model_dim, num_heads),
      pre_ffn_norm_(model_dim),
      feed_forward_(model_dim, ffn_dim) {}

Tensor DecoderLayer::forward(const Tensor& input) const {
    if (!ready() || !input.ready()) {
        return {};
    }

    Tensor attn_input = pre_attn_norm_.forward(input);
    Tensor attn_output = attention_.forward(attn_input);
    Tensor residual_1 = add(input, attn_output);

    Tensor ffn_input = pre_ffn_norm_.forward(residual_1);
    Tensor ffn_output = feed_forward_.forward(ffn_input);
    Tensor output = add(residual_1, ffn_output);

    return output;
}

bool DecoderLayer::ready() const {
    return pre_attn_norm_.ready() && attention_.ready() && pre_ffn_norm_.ready() && feed_forward_.ready();
}

std::size_t DecoderLayer::layer_index() const {
    return layer_index_;
}

}  // namespace chapter04_attention
