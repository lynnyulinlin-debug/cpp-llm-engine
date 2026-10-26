// CUDA 执行层：实现连续 Float32 Tensor 的 contract、storage 和异步复制。
// 契约：首版只接受非零连续 shape 和 Float32，不提前承诺量化或 view。
// 验证：cpp_llm_engine_cuda_execution_context_test。
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <limits>
#include <utility>

namespace cpp_llm_engine::cuda {

core::Status CudaTensor::create_contiguous(
    const std::vector<std::size_t>& shape,
    core::DType dtype,
    CudaTensor* output) {
    if (output == nullptr) return core::Status::error("CudaTensor output is null");
    if (shape.empty()) return core::Status::error("CudaTensor shape is empty");
    if (dtype != core::DType::Float32) {
        return core::Status::error("CudaTensor currently supports Float32 only");
    }

    std::size_t elements = 1;
    for (const std::size_t dimension : shape) {
        if (dimension == 0) return core::Status::error("CudaTensor dimension is zero");
        if (elements > std::numeric_limits<std::size_t>::max() / dimension) {
            return core::Status::error("CudaTensor element count overflows");
        }
        elements *= dimension;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return core::Status::error("CudaTensor byte count overflows");
    }

    CudaTensor replacement;
    replacement.contract_.shape.dimensions = shape;
    replacement.contract_.strides.resize(shape.size());
    std::size_t stride = 1;
    for (std::size_t axis = shape.size(); axis-- > 0;) {
        replacement.contract_.strides[axis] = stride;
        stride *= shape[axis];
    }
    replacement.contract_.dtype = dtype;
    replacement.contract_.device = core::DeviceKind::CUDA;
    replacement.contract_.storage_bytes = elements * sizeof(float);
    const CudaStatus allocation = DeviceBuffer::allocate(
        replacement.contract_.storage_bytes, &replacement.storage_);
    if (!allocation.is_ok()) return allocation.to_core_status();
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaTensor::copy_from_host_async(
    const void* source,
    std::size_t bytes,
    const CudaStream& stream) {
    if (bytes != size_bytes()) {
        return core::Status::error("CudaTensor H2D byte count mismatch");
    }
    return storage_.copy_from_host_async(source, bytes, stream).to_core_status();
}

core::Status CudaTensor::copy_to_host_async(
    void* destination,
    std::size_t bytes,
    const CudaStream& stream) const {
    if (bytes != size_bytes()) {
        return core::Status::error("CudaTensor D2H byte count mismatch");
    }
    return storage_.copy_to_host_async(destination, bytes, stream).to_core_status();
}

core::Status CudaTensor::reset() {
    const CudaStatus status = storage_.reset();
    contract_ = {};
    return status.to_core_status();
}

const core::TensorContract& CudaTensor::contract() const { return contract_; }

void* CudaTensor::data() { return storage_.data(); }

const void* CudaTensor::data() const { return storage_.data(); }

std::size_t CudaTensor::size_bytes() const { return storage_.size_bytes(); }

bool CudaTensor::empty() const { return storage_.empty(); }

}  // namespace cpp_llm_engine::cuda
