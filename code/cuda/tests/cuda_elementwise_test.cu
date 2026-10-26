// CUDA Backend：验证首个 elementwise backend 的正确性、边界和设备计时。
// 职责：比较 CPU golden 与 CUDA Vector Add，并覆盖非整除 grid 和错误参数。
// 通过条件：全部 CHECK 成立，且输出固定 workload 的多轮 kernel mean。
#include "cpp_llm_engine/core/elementwise_contract.h"
#include "cpp_llm_engine/cuda/cuda_event.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/device_buffer.h"
#include "cpp_llm_engine/cuda/elementwise_backend.h"
#include "cpp_llm_engine/cuda/pinned_host_buffer.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "check failed: " #condition << " at line " << __LINE__ \
                      << '\n';                                                  \
            return false;                                                       \
        }                                                                       \
    } while (false)

namespace {

using cpp_llm_engine::core::VectorAddRequest;
using cpp_llm_engine::cuda::CudaEvent;
using cpp_llm_engine::cuda::CudaStream;
using cpp_llm_engine::cuda::DeviceBuffer;
using cpp_llm_engine::cuda::PinnedHostBuffer;

bool run_correctness_case(std::size_t elements) {
    const std::size_t bytes = elements * sizeof(float);
    PinnedHostBuffer host_lhs;
    PinnedHostBuffer host_rhs;
    PinnedHostBuffer host_output;
    DeviceBuffer device_lhs;
    DeviceBuffer device_rhs;
    DeviceBuffer device_output;
    CudaStream stream;

    CHECK(PinnedHostBuffer::allocate(bytes, &host_lhs).is_ok());
    CHECK(PinnedHostBuffer::allocate(bytes, &host_rhs).is_ok());
    CHECK(PinnedHostBuffer::allocate(bytes, &host_output).is_ok());
    CHECK(DeviceBuffer::allocate(bytes, &device_lhs).is_ok());
    CHECK(DeviceBuffer::allocate(bytes, &device_rhs).is_ok());
    CHECK(DeviceBuffer::allocate(bytes, &device_output).is_ok());
    CHECK(CudaStream::create(&stream).is_ok());

    auto* lhs = static_cast<float*>(host_lhs.data());
    auto* rhs = static_cast<float*>(host_rhs.data());
    auto* output = static_cast<float*>(host_output.data());
    for (std::size_t index = 0; index < elements; ++index) {
        lhs[index] = static_cast<float>(index) * 0.25F;
        rhs[index] = 3.0F - static_cast<float>(index) * 0.125F;
    }

    CHECK(device_lhs.copy_from_host_async(lhs, bytes, stream).is_ok());
    CHECK(device_rhs.copy_from_host_async(rhs, bytes, stream).is_ok());
    const VectorAddRequest request{
        static_cast<const float*>(device_lhs.data()),
        static_cast<const float*>(device_rhs.data()),
        static_cast<float*>(device_output.data()),
        elements};
    CHECK(cpp_llm_engine::cuda::cuda_elementwise_backend()
              .vector_add(request, stream).is_ok());
    CHECK(device_output.copy_to_host_async(output, bytes, stream).is_ok());
    CHECK(stream.synchronize().is_ok());

    for (std::size_t index = 0; index < elements; ++index) {
        const float golden = lhs[index] + rhs[index];
        CHECK(std::fabs(output[index] - golden) <= 1.0e-6F);
    }
    return true;
}

bool run_benchmark() {
    constexpr std::size_t elements = 1U << 20U;
    constexpr int warmup = 1;
    constexpr int iterations = 10;
    const std::size_t bytes = elements * sizeof(float);
    PinnedHostBuffer host_lhs;
    PinnedHostBuffer host_rhs;
    PinnedHostBuffer host_output;
    DeviceBuffer lhs;
    DeviceBuffer rhs;
    DeviceBuffer output;
    CudaStream stream;
    CudaEvent start;
    CudaEvent end;

    CHECK(PinnedHostBuffer::allocate(bytes, &host_lhs).is_ok());
    CHECK(PinnedHostBuffer::allocate(bytes, &host_rhs).is_ok());
    CHECK(PinnedHostBuffer::allocate(bytes, &host_output).is_ok());
    CHECK(DeviceBuffer::allocate(bytes, &lhs).is_ok());
    CHECK(DeviceBuffer::allocate(bytes, &rhs).is_ok());
    CHECK(DeviceBuffer::allocate(bytes, &output).is_ok());
    CHECK(CudaStream::create(&stream).is_ok());
    CHECK(CudaEvent::create(&start).is_ok());
    CHECK(CudaEvent::create(&end).is_ok());

    auto* host_lhs_data = static_cast<float*>(host_lhs.data());
    auto* host_rhs_data = static_cast<float*>(host_rhs.data());
    auto* host_output_data = static_cast<float*>(host_output.data());
    for (std::size_t index = 0; index < elements; ++index) {
        host_lhs_data[index] = 1.25F;
        host_rhs_data[index] = 2.5F;
    }
    CHECK(lhs.copy_from_host_async(host_lhs_data, bytes, stream).is_ok());
    CHECK(rhs.copy_from_host_async(host_rhs_data, bytes, stream).is_ok());

    const VectorAddRequest request{
        static_cast<const float*>(lhs.data()),
        static_cast<const float*>(rhs.data()),
        static_cast<float*>(output.data()),
        elements};
    for (int iteration = 0; iteration < warmup; ++iteration) {
        CHECK(cpp_llm_engine::cuda::cuda_elementwise_backend()
                  .vector_add(request, stream).is_ok());
    }
    CHECK(stream.synchronize().is_ok());

    CHECK(start.record(stream).is_ok());
    for (int iteration = 0; iteration < iterations; ++iteration) {
        CHECK(cpp_llm_engine::cuda::cuda_elementwise_backend()
                  .vector_add(request, stream).is_ok());
    }
    CHECK(end.record(stream).is_ok());
    CHECK(end.synchronize().is_ok());

    float elapsed_ms = 0.0F;
    CHECK(CudaEvent::elapsed_time_ms(start, end, &elapsed_ms).is_ok());
    CHECK(elapsed_ms >= 0.0F);
    CHECK(output.copy_to_host_async(host_output_data, bytes, stream).is_ok());
    CHECK(stream.synchronize().is_ok());
    for (std::size_t index = 0; index < elements; ++index) {
        CHECK(host_output_data[index] == 3.75F);
    }
    std::cout << "elementwise_elements=" << elements
              << " warmup=" << warmup
              << " iterations=" << iterations
              << " kernel_mean_ms=" << elapsed_ms / iterations << '\n';
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) return 1;

    CudaStream invalid_stream;
    const auto& backend = cpp_llm_engine::cuda::cuda_elementwise_backend();
    if (backend.kind() != cpp_llm_engine::core::BackendKind::CUDA) return 1;
    if (backend.name() != "cuda-elementwise") return 1;
    if (!backend.vector_add(VectorAddRequest{}, invalid_stream).is_ok()) return 1;

    VectorAddRequest invalid_request;
    invalid_request.elements = 1;
    if (backend.vector_add(invalid_request, invalid_stream).is_ok()) return 1;

    CudaStream valid_stream;
    if (!CudaStream::create(&valid_stream).is_ok()) return 1;
    if (backend.vector_add(invalid_request, valid_stream).is_ok()) return 1;

    for (const std::size_t size : {1U, 255U, 256U, 257U, 4099U}) {
        if (!run_correctness_case(size)) return 1;
    }
    if (!run_benchmark()) return 1;
    return 0;
}

#undef CHECK
