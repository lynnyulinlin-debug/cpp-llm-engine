// CUDA Backend：验证 cuBLAS row-major MatMul 映射、错误契约和设备计时。
// 职责：以 CPU 三重循环作为多 shape golden，并覆盖 handle/stream 生命周期。
// 通过条件：全部 CHECK 成立，且输出固定 SGEMM workload 的多轮 event mean。
#include "cpp_llm_engine/core/matmul_contract.h"
#include "cpp_llm_engine/cuda/cublas_matmul_backend.h"
#include "cpp_llm_engine/cuda/cublas_status.h"
#include "cpp_llm_engine/cuda/cuda_event.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/device_buffer.h"
#include "cpp_llm_engine/cuda/pinned_host_buffer.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "check failed: " #condition << " at line " << __LINE__ \
                      << '\n';                                                  \
            return false;                                                       \
        }                                                                       \
    } while (false)

namespace {

using cpp_llm_engine::core::MatmulRequest;
using cpp_llm_engine::cuda::CublasMatmulBackend;
using cpp_llm_engine::cuda::CudaEvent;
using cpp_llm_engine::cuda::CudaStream;
using cpp_llm_engine::cuda::DeviceBuffer;
using cpp_llm_engine::cuda::PinnedHostBuffer;

void cpu_matmul(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t rows,
    std::size_t inner,
    std::size_t cols) {
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            float value = 0.0F;
            for (std::size_t index = 0; index < inner; ++index) {
                value += lhs[row * inner + index] * rhs[index * cols + col];
            }
            output[row * cols + col] = value;
        }
    }
}

bool run_correctness_case(
    std::size_t rows,
    std::size_t inner,
    std::size_t cols,
    const CublasMatmulBackend& backend,
    const CudaStream& stream) {
    const std::size_t lhs_bytes = rows * inner * sizeof(float);
    const std::size_t rhs_bytes = inner * cols * sizeof(float);
    const std::size_t output_bytes = rows * cols * sizeof(float);
    PinnedHostBuffer host_lhs;
    PinnedHostBuffer host_rhs;
    PinnedHostBuffer host_output;
    PinnedHostBuffer host_golden;
    DeviceBuffer device_lhs;
    DeviceBuffer device_rhs;
    DeviceBuffer device_output;

    CHECK(PinnedHostBuffer::allocate(lhs_bytes, &host_lhs).is_ok());
    CHECK(PinnedHostBuffer::allocate(rhs_bytes, &host_rhs).is_ok());
    CHECK(PinnedHostBuffer::allocate(output_bytes, &host_output).is_ok());
    CHECK(PinnedHostBuffer::allocate(output_bytes, &host_golden).is_ok());
    CHECK(DeviceBuffer::allocate(lhs_bytes, &device_lhs).is_ok());
    CHECK(DeviceBuffer::allocate(rhs_bytes, &device_rhs).is_ok());
    CHECK(DeviceBuffer::allocate(output_bytes, &device_output).is_ok());

    auto* lhs = static_cast<float*>(host_lhs.data());
    auto* rhs = static_cast<float*>(host_rhs.data());
    auto* output = static_cast<float*>(host_output.data());
    auto* golden = static_cast<float*>(host_golden.data());
    for (std::size_t index = 0; index < rows * inner; ++index) {
        lhs[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.125F;
    }
    for (std::size_t index = 0; index < inner * cols; ++index) {
        rhs[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.25F;
    }
    cpu_matmul(lhs, rhs, golden, rows, inner, cols);

    CHECK(device_lhs.copy_from_host_async(lhs, lhs_bytes, stream).is_ok());
    CHECK(device_rhs.copy_from_host_async(rhs, rhs_bytes, stream).is_ok());
    const MatmulRequest request{
        static_cast<const float*>(device_lhs.data()),
        static_cast<const float*>(device_rhs.data()),
        static_cast<float*>(device_output.data()),
        rows,
        inner,
        cols};
    CHECK(backend.compute(request, stream).is_ok());
    CHECK(device_output.copy_to_host_async(output, output_bytes, stream).is_ok());
    CHECK(stream.synchronize().is_ok());

    for (std::size_t index = 0; index < rows * cols; ++index) {
        const float tolerance = 1.0e-4F * std::max(1.0F, std::fabs(golden[index]));
        CHECK(std::fabs(output[index] - golden[index]) <= tolerance);
    }
    return true;
}

bool run_benchmark(
    const CublasMatmulBackend& backend,
    const CudaStream& stream) {
    constexpr std::size_t rows = 256;
    constexpr std::size_t inner = 256;
    constexpr std::size_t cols = 256;
    constexpr int warmup = 1;
    constexpr int iterations = 10;
    constexpr std::size_t matrix_elements = 256U * 256U;
    constexpr std::size_t bytes = matrix_elements * sizeof(float);
    PinnedHostBuffer host_lhs;
    PinnedHostBuffer host_rhs;
    PinnedHostBuffer host_output;
    DeviceBuffer lhs;
    DeviceBuffer rhs;
    DeviceBuffer output;
    CudaEvent start;
    CudaEvent end;

    CHECK(PinnedHostBuffer::allocate(bytes, &host_lhs).is_ok());
    CHECK(PinnedHostBuffer::allocate(bytes, &host_rhs).is_ok());
    CHECK(PinnedHostBuffer::allocate(bytes, &host_output).is_ok());
    CHECK(DeviceBuffer::allocate(bytes, &lhs).is_ok());
    CHECK(DeviceBuffer::allocate(bytes, &rhs).is_ok());
    CHECK(DeviceBuffer::allocate(bytes, &output).is_ok());
    CHECK(CudaEvent::create(&start).is_ok());
    CHECK(CudaEvent::create(&end).is_ok());

    auto* host_lhs_data = static_cast<float*>(host_lhs.data());
    auto* host_rhs_data = static_cast<float*>(host_rhs.data());
    auto* host_output_data = static_cast<float*>(host_output.data());
    std::fill_n(host_lhs_data, matrix_elements, 0.5F);
    std::fill_n(host_rhs_data, matrix_elements, 0.25F);
    CHECK(lhs.copy_from_host_async(host_lhs_data, bytes, stream).is_ok());
    CHECK(rhs.copy_from_host_async(host_rhs_data, bytes, stream).is_ok());

    const MatmulRequest request{
        static_cast<const float*>(lhs.data()),
        static_cast<const float*>(rhs.data()),
        static_cast<float*>(output.data()),
        rows,
        inner,
        cols};
    for (int iteration = 0; iteration < warmup; ++iteration) {
        CHECK(backend.compute(request, stream).is_ok());
    }
    CHECK(stream.synchronize().is_ok());

    CHECK(start.record(stream).is_ok());
    for (int iteration = 0; iteration < iterations; ++iteration) {
        CHECK(backend.compute(request, stream).is_ok());
    }
    CHECK(end.record(stream).is_ok());
    CHECK(end.synchronize().is_ok());

    float elapsed_ms = 0.0F;
    CHECK(CudaEvent::elapsed_time_ms(start, end, &elapsed_ms).is_ok());
    CHECK(output.copy_to_host_async(host_output_data, bytes, stream).is_ok());
    CHECK(stream.synchronize().is_ok());
    for (std::size_t index = 0; index < matrix_elements; ++index) {
        CHECK(std::fabs(host_output_data[index] - 32.0F) <= 1.0e-4F);
    }
    std::cout << "cublas_m=" << rows
              << " cublas_k=" << inner
              << " cublas_n=" << cols
              << " warmup=" << warmup
              << " iterations=" << iterations
              << " sgemm_mean_ms=" << elapsed_ms / iterations << '\n';
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) return 1;

    const auto synthetic = cpp_llm_engine::cuda::CublasStatus::from(
        CUBLAS_STATUS_INVALID_VALUE, "synthetic");
    if (synthetic.is_ok() || synthetic.code() != CUBLAS_STATUS_INVALID_VALUE ||
        synthetic.operation() != "synthetic" ||
        synthetic.message().find("synthetic") == std::string::npos ||
        synthetic.to_core_status().is_ok()) {
        return 1;
    }
    if (CublasMatmulBackend::create(nullptr).is_ok()) return 1;

    CublasMatmulBackend invalid_backend;
    CudaStream invalid_stream;
    if (invalid_backend.compute(MatmulRequest{}, invalid_stream).is_ok()) return 1;

    CublasMatmulBackend backend;
    CudaStream stream;
    if (!CublasMatmulBackend::create(&backend).is_ok()) return 1;
    if (!CudaStream::create(&stream).is_ok()) return 1;
    if (backend.kind() != cpp_llm_engine::core::BackendKind::CUDA ||
        backend.name() != "cublas-sgemm") {
        return 1;
    }
    if (backend.compute(MatmulRequest{}, stream).is_ok()) return 1;

    MatmulRequest too_large;
    too_large.lhs = reinterpret_cast<const float*>(1);
    too_large.rhs = reinterpret_cast<const float*>(1);
    too_large.output = reinterpret_cast<float*>(1);
    too_large.rows = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
    too_large.inner = 1;
    too_large.cols = 1;
    if (backend.compute(too_large, stream).is_ok()) return 1;

    for (const auto shape : {
             std::array<std::size_t, 3>{1, 1, 1},
             std::array<std::size_t, 3>{2, 3, 4},
             std::array<std::size_t, 3>{7, 5, 9},
             std::array<std::size_t, 3>{31, 33, 17}}) {
        if (!run_correctness_case(shape[0], shape[1], shape[2], backend, stream)) {
            return 1;
        }
    }

    CublasMatmulBackend moved(std::move(backend));
    if (backend.valid() || !moved.valid()) return 1;
    if (!run_benchmark(moved, stream)) return 1;
    if (!moved.reset().is_ok() || !moved.reset().is_ok()) return 1;
    return 0;
}

#undef CHECK
