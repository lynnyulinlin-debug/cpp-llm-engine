#pragma once

#include "chapter04_attention/tensor.h"

#include <cstddef>
#include <vector>

namespace chapter04_attention {

class LayerNorm final {
public:
    explicit LayerNorm(std::size_t dim = 0, float epsilon = 1e-5f);

    Tensor forward(const Tensor& input) const;
    bool ready() const;

    std::size_t dim() const;
    float epsilon() const;

private:
    std::size_t dim_ = 0;
    float epsilon_ = 1e-5f;
    std::vector<float> gamma_;
    std::vector<float> beta_;
};

}  // namespace chapter04_attention
