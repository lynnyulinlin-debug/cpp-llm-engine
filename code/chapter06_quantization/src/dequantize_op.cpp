#include "chapter06_quantization/dequantize_op.h"

namespace chapter06_quantization {

Tensor DequantizeOp::forward(const Tensor& input) const {
    return input.dequantize();
}

}  // namespace chapter06_quantization
