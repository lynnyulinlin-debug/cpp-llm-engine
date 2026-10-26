// CUDA 基础层：实现 event 的独占所有权、记录、同步与设备计时。
// 契约：record 只接受有效 stream；elapsed_time_ms 只接受两个有效 event。
// 验证：cpp_llm_engine_cuda_test 对完整异步流水线进行 event 计时。
#include "cpp_llm_engine/cuda/cuda_event.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <utility>

namespace cpp_llm_engine::cuda {

CudaEvent::~CudaEvent() { release_noexcept(); }

CudaEvent::CudaEvent(CudaEvent&& other) noexcept
    : event_(std::exchange(other.event_, nullptr)) {}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
    if (this == &other) return *this;
    release_noexcept();
    event_ = std::exchange(other.event_, nullptr);
    return *this;
}

CudaStatus CudaEvent::create(CudaEvent* output, unsigned int flags) {
    if (output == nullptr) {
        return CudaStatus::from(cudaErrorInvalidValue, "CudaEvent::create output");
    }
    CudaEvent replacement;
    const CudaStatus status = CudaStatus::from(
        cudaEventCreateWithFlags(&replacement.event_, flags),
        "cudaEventCreateWithFlags");
    if (!status.is_ok()) return status;
    *output = std::move(replacement);
    return CudaStatus::success();
}

CudaStatus CudaEvent::elapsed_time_ms(
    const CudaEvent& start,
    const CudaEvent& end,
    float* milliseconds) {
    if (milliseconds == nullptr || !start.valid() || !end.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle, "CudaEvent::elapsed_time_ms");
    }
    return CudaStatus::from(
        cudaEventElapsedTime(milliseconds, start.event_, end.event_),
        "cudaEventElapsedTime");
}

CudaStatus CudaEvent::record(const CudaStream& stream) {
    if (!valid() || !stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidResourceHandle, "CudaEvent::record");
    }
    return CudaStatus::from(cudaEventRecord(event_, stream.native_handle()), "cudaEventRecord");
}

CudaStatus CudaEvent::synchronize() const {
    if (!valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle, "CudaEvent::synchronize");
    }
    return CudaStatus::from(cudaEventSynchronize(event_), "cudaEventSynchronize");
}

CudaStatus CudaEvent::reset() {
    if (!valid()) return CudaStatus::success();
    const cudaEvent_t event = event_;
    event_ = nullptr;
    return CudaStatus::from(cudaEventDestroy(event), "cudaEventDestroy");
}

cudaEvent_t CudaEvent::native_handle() const { return event_; }

bool CudaEvent::valid() const { return event_ != nullptr; }

void CudaEvent::release_noexcept() noexcept {
    if (event_ != nullptr) (void)cudaEventDestroy(event_);
    event_ = nullptr;
}

}  // namespace cpp_llm_engine::cuda
