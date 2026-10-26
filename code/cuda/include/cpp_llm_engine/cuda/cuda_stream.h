// CUDA 基础层：声明 move-only 的非阻塞 stream 所有权对象。
// 职责：封装 cudaStreamCreateWithFlags、同步和显式销毁。
// 验证：真实 GPU 测试覆盖创建、异步流水线、移动和重复释放。
#pragma once

#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cuda_runtime_api.h>

namespace cpp_llm_engine::cuda {

class CudaStream final {
public:
    CudaStream() = default;
    ~CudaStream();

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    static CudaStatus create(
        CudaStream* output,
        unsigned int flags = cudaStreamNonBlocking);

    CudaStatus synchronize() const;
    CudaStatus reset();
    cudaStream_t native_handle() const;
    bool valid() const;

private:
    void release_noexcept() noexcept;

    cudaStream_t stream_ = nullptr;
};

}  // namespace cpp_llm_engine::cuda
