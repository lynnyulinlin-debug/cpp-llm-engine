#pragma once

#include "chapter06_quantization/tensor.h"

#include <cstddef>
#include <vector>

namespace chapter06_quantization {

class LinearLayer final {
public:
    LinearLayer(std::size_t in_features = 0, std::size_t out_features = 0);

    Tensor forward(const Tensor& input) const;
    bool ready() const;

private:
    std::size_t in_features_ = 0;
    std::size_t out_features_ = 0;
    Tensor weight_;
    std::vector<float> bias_;
};

}  // namespace chapter06_quantization
