#pragma once

#include "chapter04_attention/tensor.h"

#include <cstddef>
#include <vector>

namespace chapter04_attention {

class LinearLayer final {
public:
    LinearLayer(std::size_t in_features = 0, std::size_t out_features = 0);

    Tensor forward(const Tensor& input) const;
    bool ready() const;

    std::size_t in_features() const;
    std::size_t out_features() const;

private:
    std::size_t in_features_ = 0;
    std::size_t out_features_ = 0;
    Tensor weight_;
    std::vector<float> bias_;
};

}  // namespace chapter04_attention
