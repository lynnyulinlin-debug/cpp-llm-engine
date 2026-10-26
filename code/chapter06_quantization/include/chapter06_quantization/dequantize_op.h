// Chapter 06：声明教学 INT8 Tensor 的反量化算子。
// 契约：输入必须携带量化 scale，输出为同形状浮点 Tensor。
#pragma once

#include "chapter06_quantization/tensor.h"

namespace chapter06_quantization {

class DequantizeOp final {
public:
    Tensor forward(const Tensor& input) const;
};

}  // namespace chapter06_quantization
