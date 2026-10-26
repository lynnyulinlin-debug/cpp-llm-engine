// CUDA Backend：实现 Float32 Vector Add 的参数边界和 kernel launch。
// 契约：零元素请求是 no-op；非零请求要求三个 device 指针和有效 stream。
// 验证：cpp_llm_engine_cuda_elementwise_test 与 CPU reference 对照。
#include "cpp_llm_engine/cuda/elementwise_backend.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>

#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

constexpr std::size_t kBlockSize = 256;

__global__ void vector_add_kernel(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t elements) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = lhs[index] + rhs[index];
}

}  // namespace

core::BackendKind CudaElementwiseBackend::kind() const {
    return core::BackendKind::CUDA;
}

std::string_view CudaElementwiseBackend::name() const {
    return "cuda-elementwise";
}

CudaStatus CudaElementwiseBackend::vector_add(
    const core::VectorAddRequest& request,
    const CudaStream& stream) const {
    if (request.elements == 0) return CudaStatus::success();
    if (request.lhs == nullptr || request.rhs == nullptr || request.output == nullptr) {
        return CudaStatus::from(
            cudaErrorInvalidDevicePointer,
            "CudaElementwiseBackend::vector_add pointers");
    }
    if (!stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle,
            "CudaElementwiseBackend::vector_add stream");
    }

    const std::size_t blocks = request.elements / kBlockSize +
        (request.elements % kBlockSize != 0 ? 1 : 0);
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(
            cudaErrorInvalidConfiguration,
            "CudaElementwiseBackend::vector_add grid");
    }

    vector_add_kernel<<<
        static_cast<unsigned int>(blocks),
        static_cast<unsigned int>(kBlockSize),
        0,
        stream.native_handle()>>>(
            request.lhs, request.rhs, request.output, request.elements);
    return check_last_cuda_error("vector_add_kernel launch");
}

const CudaElementwiseBackend& cuda_elementwise_backend() {
    static const CudaElementwiseBackend backend;
    return backend;
}

}  // namespace cpp_llm_engine::cuda
