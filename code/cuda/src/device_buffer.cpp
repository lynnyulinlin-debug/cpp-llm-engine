// CUDA 基础层：实现设备内存所有权和同步、异步 host/device 复制。
// 契约：零长度缓冲区合法；复制先检查空指针、offset 溢出和容量边界。
// 验证：cpp_llm_engine_cuda_test 在真实设备上执行复制、kernel 和释放。
#include "cpp_llm_engine/cuda/device_buffer.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <utility>

namespace cpp_llm_engine::cuda {

DeviceBuffer::~DeviceBuffer() { release_noexcept(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) return *this;
    release_noexcept();
    data_ = std::exchange(other.data_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
    return *this;
}

CudaStatus DeviceBuffer::allocate(std::size_t bytes, DeviceBuffer* output) {
    if (output == nullptr) {
        return CudaStatus::from(cudaErrorInvalidValue, "DeviceBuffer::allocate output");
    }

    DeviceBuffer replacement;
    if (bytes > 0) {
        void* allocation = nullptr;
        const CudaStatus status = CudaStatus::from(
            cudaMalloc(&allocation, bytes), "cudaMalloc DeviceBuffer");
        if (!status.is_ok()) return status;
        replacement.data_ = allocation;
        replacement.size_bytes_ = bytes;
    }
    *output = std::move(replacement);
    return CudaStatus::success();
}

CudaStatus DeviceBuffer::validate_copy(
    const void* host_pointer,
    std::size_t bytes,
    std::size_t offset,
    std::size_t capacity,
    const char* operation) {
    if (bytes == 0) return CudaStatus::success();
    if (host_pointer == nullptr || offset > capacity || bytes > capacity - offset) {
        return CudaStatus::from(cudaErrorInvalidValue, operation);
    }
    return CudaStatus::success();
}

CudaStatus DeviceBuffer::copy_from_host(
    const void* source,
    std::size_t bytes,
    std::size_t destination_offset) {
    const CudaStatus validation = validate_copy(
        source, bytes, destination_offset, size_bytes_, "DeviceBuffer::copy_from_host");
    if (!validation.is_ok() || bytes == 0) return validation;
    auto* destination = static_cast<unsigned char*>(data_) + destination_offset;
    return CudaStatus::from(
        cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
}

CudaStatus DeviceBuffer::copy_to_host(
    void* destination,
    std::size_t bytes,
    std::size_t source_offset) const {
    const CudaStatus validation = validate_copy(
        destination, bytes, source_offset, size_bytes_, "DeviceBuffer::copy_to_host");
    if (!validation.is_ok() || bytes == 0) return validation;
    const auto* source = static_cast<const unsigned char*>(data_) + source_offset;
    return CudaStatus::from(
        cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");
}

CudaStatus DeviceBuffer::copy_from_host_async(
    const void* source,
    std::size_t bytes,
    const CudaStream& stream,
    std::size_t destination_offset) {
    const CudaStatus validation = validate_copy(
        source, bytes, destination_offset, size_bytes_,
        "DeviceBuffer::copy_from_host_async");
    if (!validation.is_ok() || bytes == 0) return validation;
    if (!stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle,
            "DeviceBuffer::copy_from_host_async stream");
    }
    auto* destination = static_cast<unsigned char*>(data_) + destination_offset;
    return CudaStatus::from(
        cudaMemcpyAsync(
            destination, source, bytes, cudaMemcpyHostToDevice,
            stream.native_handle()),
        "cudaMemcpyAsync H2D");
}

CudaStatus DeviceBuffer::copy_to_host_async(
    void* destination,
    std::size_t bytes,
    const CudaStream& stream,
    std::size_t source_offset) const {
    const CudaStatus validation = validate_copy(
        destination, bytes, source_offset, size_bytes_,
        "DeviceBuffer::copy_to_host_async");
    if (!validation.is_ok() || bytes == 0) return validation;
    if (!stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle,
            "DeviceBuffer::copy_to_host_async stream");
    }
    const auto* source = static_cast<const unsigned char*>(data_) + source_offset;
    return CudaStatus::from(
        cudaMemcpyAsync(
            destination, source, bytes, cudaMemcpyDeviceToHost,
            stream.native_handle()),
        "cudaMemcpyAsync D2H");
}

CudaStatus DeviceBuffer::reset() {
    if (data_ == nullptr) {
        size_bytes_ = 0;
        return CudaStatus::success();
    }
    void* allocation = data_;
    data_ = nullptr;
    size_bytes_ = 0;
    return CudaStatus::from(cudaFree(allocation), "cudaFree DeviceBuffer");
}

void* DeviceBuffer::data() { return data_; }

const void* DeviceBuffer::data() const { return data_; }

std::size_t DeviceBuffer::size_bytes() const { return size_bytes_; }

bool DeviceBuffer::empty() const { return size_bytes_ == 0; }

void DeviceBuffer::release_noexcept() noexcept {
    if (data_ != nullptr) (void)cudaFree(data_);
    data_ = nullptr;
    size_bytes_ = 0;
}

}  // namespace cpp_llm_engine::cuda
