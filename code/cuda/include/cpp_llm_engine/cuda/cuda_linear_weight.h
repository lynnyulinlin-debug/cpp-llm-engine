// CUDA 执行层：统一 Float32 与量化 Linear 权重的只读视图。
// 职责：向 Decoder 隐藏具体 storage 类型，同时保留输入/输出宽度语义。
// 契约：Float32 shape 为 [input,output]；量化矩阵为 rows=output、cols=input。
// 验证：合成 Decoder 同时运行 Float32 与 Q4_K/Q6_K 权重路径。
#pragma once

#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class CudaLinearWeight final {
public:
    enum class Kind { Invalid, Float32, Quantized };

    static CudaLinearWeight float32(const CudaTensor* tensor) {
        CudaLinearWeight result;
        result.kind_ = Kind::Float32;
        result.float32_ = tensor;
        return result;
    }
    static CudaLinearWeight quantized(const CudaQuantizedMatrix* matrix) {
        CudaLinearWeight result;
        result.kind_ = Kind::Quantized;
        result.quantized_ = matrix;
        return result;
    }

    Kind kind() const { return kind_; }
    const CudaTensor* float32_tensor() const { return float32_; }
    const CudaQuantizedMatrix* quantized_matrix() const { return quantized_; }
    bool valid() const {
        return (kind_ == Kind::Float32 && float32_ != nullptr && !float32_->empty()) ||
               (kind_ == Kind::Quantized && quantized_ != nullptr && !quantized_->empty());
    }

private:
    Kind kind_ = Kind::Invalid;
    const CudaTensor* float32_ = nullptr;
    const CudaQuantizedMatrix* quantized_ = nullptr;
};

}  // namespace cpp_llm_engine::cuda
