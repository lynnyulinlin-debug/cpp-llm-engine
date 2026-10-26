// CUDA 执行层：声明拥有 DeviceBuffer 的连续 CUDA Tensor。
// 职责：把 shape、stride、dtype、device contract 与设备 storage 绑定。
// 验证：ExecutionContext CTest 覆盖创建、复制、移动和非法 contract。
#pragma once

#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/device_buffer.h"

#include <cstddef>
#include <vector>

namespace cpp_llm_engine::cuda {

class CudaStream;

class CudaTensor final {
public:
    CudaTensor() = default;
    ~CudaTensor() = default;

    CudaTensor(const CudaTensor&) = delete;
    CudaTensor& operator=(const CudaTensor&) = delete;
    CudaTensor(CudaTensor&&) noexcept = default;
    CudaTensor& operator=(CudaTensor&&) noexcept = default;

    static core::Status create_contiguous(
        const std::vector<std::size_t>& shape,
        core::DType dtype,
        CudaTensor* output);

    core::Status copy_from_host_async(
        const void* source,
        std::size_t bytes,
        const CudaStream& stream);
    core::Status copy_to_host_async(
        void* destination,
        std::size_t bytes,
        const CudaStream& stream) const;
    core::Status reset();

    const core::TensorContract& contract() const;
    void* data();
    const void* data() const;
    std::size_t size_bytes() const;
    bool empty() const;

private:
    core::TensorContract contract_;
    DeviceBuffer storage_;
};

}  // namespace cpp_llm_engine::cuda
