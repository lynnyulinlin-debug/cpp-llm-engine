// CUDA 执行层：声明可增长复用的设备 workspace。
// 职责：集中临时显存所有权，避免算子内部反复 cudaMalloc/cudaFree。
// 验证：CTest 覆盖首次分配、小请求复用、增长和移动所有权。
#pragma once

#include "cpp_llm_engine/cuda/device_buffer.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class CudaWorkspace final {
public:
    CudaWorkspace() = default;
    ~CudaWorkspace() = default;

    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;
    CudaWorkspace(CudaWorkspace&& other) noexcept;
    CudaWorkspace& operator=(CudaWorkspace&& other) noexcept;

    CudaStatus reserve(std::size_t bytes);
    CudaStatus reset();
    void* data();
    const void* data() const;
    std::size_t capacity_bytes() const;
    std::size_t allocation_count() const;

private:
    DeviceBuffer storage_;
    std::size_t allocation_count_ = 0;
};

}  // namespace cpp_llm_engine::cuda
