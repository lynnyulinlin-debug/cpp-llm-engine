// CUDA 基础层：声明 move-only 的页锁定 host 字节缓冲区。
// 职责：为可与 kernel 重叠的 cudaMemcpyAsync 提供稳定的 host 内存。
// 验证：真实 GPU 测试覆盖零长度、异步往返复制、移动和显式释放。
#pragma once

#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class PinnedHostBuffer final {
public:
    PinnedHostBuffer() = default;
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    static CudaStatus allocate(std::size_t bytes, PinnedHostBuffer* output);

    CudaStatus reset();
    void* data();
    const void* data() const;
    std::size_t size_bytes() const;
    bool empty() const;

private:
    void release_noexcept() noexcept;

    void* data_ = nullptr;
    std::size_t size_bytes_ = 0;
};

}  // namespace cpp_llm_engine::cuda
