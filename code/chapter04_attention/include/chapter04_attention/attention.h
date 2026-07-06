#pragma once

#include "chapter04_attention/linear_layer.h"

#include <cstddef>

namespace chapter04_attention {

class MultiHeadAttention final {
public:
    MultiHeadAttention(std::size_t model_dim = 0, std::size_t num_heads = 0);

    Tensor forward(const Tensor& input) const;
    bool ready() const;

    std::size_t model_dim() const;
    std::size_t num_heads() const;

private:
    std::size_t model_dim_ = 0;
    std::size_t num_heads_ = 0;
    LinearLayer q_proj_;
    LinearLayer k_proj_;
    LinearLayer v_proj_;
    LinearLayer out_proj_;
};

}  // namespace chapter04_attention
