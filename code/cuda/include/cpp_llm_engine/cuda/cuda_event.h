// CUDA 基础层：声明 move-only 的 event 所有权和设备计时接口。
// 职责：在指定 stream 记录事件，并计算两个事件之间的设备时间。
// 验证：真实 GPU 测试覆盖记录、同步、计时、移动和错误句柄。
#pragma once

#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cuda_runtime_api.h>

namespace cpp_llm_engine::cuda {

class CudaStream;

class CudaEvent final {
public:
    CudaEvent() = default;
    ~CudaEvent();

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;

    static CudaStatus create(
        CudaEvent* output,
        unsigned int flags = cudaEventDefault);
    static CudaStatus elapsed_time_ms(
        const CudaEvent& start,
        const CudaEvent& end,
        float* milliseconds);

    CudaStatus record(const CudaStream& stream);
    CudaStatus synchronize() const;
    CudaStatus reset();
    cudaEvent_t native_handle() const;
    bool valid() const;

private:
    void release_noexcept() noexcept;

    cudaEvent_t event_ = nullptr;
};

}  // namespace cpp_llm_engine::cuda
