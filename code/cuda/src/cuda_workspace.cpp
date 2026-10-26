// CUDA 执行层：实现只在容量不足时增长的设备 workspace。
// 契约：reserve 不缩容；成功增长后替换旧 allocation 并递增计数。
// 验证：cpp_llm_engine_cuda_execution_context_test。
#include "cpp_llm_engine/cuda/cuda_workspace.h"

#include <utility>

namespace cpp_llm_engine::cuda {

CudaWorkspace::CudaWorkspace(CudaWorkspace&& other) noexcept
    : storage_(std::move(other.storage_)),
      allocation_count_(std::exchange(other.allocation_count_, 0)) {}

CudaWorkspace& CudaWorkspace::operator=(CudaWorkspace&& other) noexcept {
    if (this == &other) return *this;
    storage_ = std::move(other.storage_);
    allocation_count_ = std::exchange(other.allocation_count_, 0);
    return *this;
}

CudaStatus CudaWorkspace::reserve(std::size_t bytes) {
    if (bytes <= capacity_bytes()) return CudaStatus::success();
    DeviceBuffer replacement;
    const CudaStatus status = DeviceBuffer::allocate(bytes, &replacement);
    if (!status.is_ok()) return status;
    const CudaStatus release_status = storage_.reset();
    if (!release_status.is_ok()) return release_status;
    storage_ = std::move(replacement);
    ++allocation_count_;
    return CudaStatus::success();
}

CudaStatus CudaWorkspace::reset() {
    const CudaStatus status = storage_.reset();
    allocation_count_ = 0;
    return status;
}

void* CudaWorkspace::data() { return storage_.data(); }

const void* CudaWorkspace::data() const { return storage_.data(); }

std::size_t CudaWorkspace::capacity_bytes() const { return storage_.size_bytes(); }

std::size_t CudaWorkspace::allocation_count() const { return allocation_count_; }

}  // namespace cpp_llm_engine::cuda
