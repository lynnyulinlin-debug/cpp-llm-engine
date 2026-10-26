// CUDA 基础层：声明 move-only 的设备字节缓冲区。
// 职责：集中管理 cudaMalloc/cudaFree，并校验同步及异步 H2D/D2H 的范围。
// 验证：真实 GPU 测试覆盖零长度、复制、kernel、移动、越界和释放。
#pragma once

#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class CudaStream;

class DeviceBuffer final {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    static CudaStatus allocate(std::size_t bytes, DeviceBuffer* output);

    CudaStatus copy_from_host(
        const void* source,
        std::size_t bytes,
        std::size_t destination_offset = 0);
    CudaStatus copy_to_host(
        void* destination,
        std::size_t bytes,
        std::size_t source_offset = 0) const;
    CudaStatus copy_from_host_async(
        const void* source,
        std::size_t bytes,
        const CudaStream& stream,
        std::size_t destination_offset = 0);
    CudaStatus copy_to_host_async(
        void* destination,
        std::size_t bytes,
        const CudaStream& stream,
        std::size_t source_offset = 0) const;
    CudaStatus reset();

    void* data();
    const void* data() const;
    std::size_t size_bytes() const;
    bool empty() const;

private:
    static CudaStatus validate_copy(
        const void* host_pointer,
        std::size_t bytes,
        std::size_t offset,
        std::size_t capacity,
        const char* operation);
    void release_noexcept() noexcept;

    void* data_ = nullptr;
    std::size_t size_bytes_ = 0;
};

}  // namespace cpp_llm_engine::cuda
