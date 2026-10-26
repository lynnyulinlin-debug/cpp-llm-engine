// CUDA 基础层：实现页锁定 host 内存的独占所有权。
// 契约：零长度分配产生合法空对象；非零分配使用 cudaMallocHost。
// 验证：cpp_llm_engine_cuda_test 将该内存用于异步 H2D 和 D2H。
#include "cpp_llm_engine/cuda/pinned_host_buffer.h"

#include <cuda_runtime_api.h>

#include <utility>

namespace cpp_llm_engine::cuda {

PinnedHostBuffer::~PinnedHostBuffer() { release_noexcept(); }

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this == &other) return *this;
    release_noexcept();
    data_ = std::exchange(other.data_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
    return *this;
}

CudaStatus PinnedHostBuffer::allocate(
    std::size_t bytes,
    PinnedHostBuffer* output) {
    if (output == nullptr) {
        return CudaStatus::from(
            cudaErrorInvalidValue, "PinnedHostBuffer::allocate output");
    }
    PinnedHostBuffer replacement;
    if (bytes > 0) {
        const CudaStatus status = CudaStatus::from(
            cudaMallocHost(&replacement.data_, bytes),
            "cudaMallocHost PinnedHostBuffer");
        if (!status.is_ok()) return status;
        replacement.size_bytes_ = bytes;
    }
    *output = std::move(replacement);
    return CudaStatus::success();
}

CudaStatus PinnedHostBuffer::reset() {
    if (data_ == nullptr) {
        size_bytes_ = 0;
        return CudaStatus::success();
    }
    void* allocation = data_;
    data_ = nullptr;
    size_bytes_ = 0;
    return CudaStatus::from(cudaFreeHost(allocation), "cudaFreeHost PinnedHostBuffer");
}

void* PinnedHostBuffer::data() { return data_; }

const void* PinnedHostBuffer::data() const { return data_; }

std::size_t PinnedHostBuffer::size_bytes() const { return size_bytes_; }

bool PinnedHostBuffer::empty() const { return size_bytes_ == 0; }

void PinnedHostBuffer::release_noexcept() noexcept {
    if (data_ != nullptr) (void)cudaFreeHost(data_);
    data_ = nullptr;
    size_bytes_ = 0;
}

}  // namespace cpp_llm_engine::cuda
