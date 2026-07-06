#pragma once

#include "chapter06_quantization/tensor.h"

namespace chapter06_quantization {

class DequantizeOp final {
public:
    Tensor forward(const Tensor& input) const;
};

}  // namespace chapter06_quantization
