// CUDA 基础层：实现 stream 的独占所有权和同步契约。
// 契约：默认对象无句柄；create 成功后由对象独占；移动后源对象失效。
// 验证：cpp_llm_engine_cuda_test 使用同一 stream 串联复制和 kernel。
#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <utility>

namespace cpp_llm_engine::cuda {

CudaStream::~CudaStream() { release_noexcept(); }

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this == &other) return *this;
    release_noexcept();
    stream_ = std::exchange(other.stream_, nullptr);
    return *this;
}

CudaStatus CudaStream::create(CudaStream* output, unsigned int flags) {
    if (output == nullptr) {
        return CudaStatus::from(cudaErrorInvalidValue, "CudaStream::create output");
    }
    CudaStream replacement;
    const CudaStatus status = CudaStatus::from(
        cudaStreamCreateWithFlags(&replacement.stream_, flags),
        "cudaStreamCreateWithFlags");
    if (!status.is_ok()) return status;
    *output = std::move(replacement);
    return CudaStatus::success();
}

CudaStatus CudaStream::synchronize() const {
    if (!valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle, "CudaStream::synchronize");
    }
    return CudaStatus::from(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

CudaStatus CudaStream::reset() {
    if (!valid()) return CudaStatus::success();
    const cudaStream_t stream = stream_;
    stream_ = nullptr;
    return CudaStatus::from(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

cudaStream_t CudaStream::native_handle() const { return stream_; }

bool CudaStream::valid() const { return stream_ != nullptr; }

void CudaStream::release_noexcept() noexcept {
    if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    stream_ = nullptr;
}

}  // namespace cpp_llm_engine::cuda
