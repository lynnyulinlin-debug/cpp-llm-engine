// CUDA 基础层：真实设备上的错误、资源所有权和异步流水线测试。
// 职责：验证内存、stream、event、H2D、kernel、D2H、计时和错误路径。
// 通过条件：所有 CHECK 成立，且 CTest 在可访问 GPU 的宿主环境退出 0。
#include "cpp_llm_engine/cuda/cuda_event.h"
#include "cpp_llm_engine/cuda/cuda_status.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/device_buffer.h"
#include "cpp_llm_engine/cuda/pinned_host_buffer.h"

#include <cuda_runtime.h>

#include <array>
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
            return 1;                                                           \
        }                                                                       \
    } while (false)

namespace {

__global__ void add_one(int* values, std::size_t size) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) values[index] += 1;
}

}  // namespace

int main() {
    using cpp_llm_engine::cuda::CudaEvent;
    using cpp_llm_engine::cuda::CudaStatus;
    using cpp_llm_engine::cuda::CudaStream;
    using cpp_llm_engine::cuda::DeviceBuffer;
    using cpp_llm_engine::cuda::PinnedHostBuffer;

    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count) == cudaSuccess);
    CHECK(device_count > 0);

    const CudaStatus synthetic = CudaStatus::from(cudaErrorInvalidValue, "synthetic");
    CHECK(!synthetic.is_ok());
    CHECK(synthetic.code() == cudaErrorInvalidValue);
    CHECK(synthetic.operation() == "synthetic");
    CHECK(synthetic.message().find("synthetic") != std::string::npos);
    CHECK(!synthetic.to_core_status().is_ok());

    DeviceBuffer zero;
    CHECK(DeviceBuffer::allocate(0, &zero).is_ok());
    CHECK(zero.empty());
    CHECK(zero.data() == nullptr);
    CHECK(zero.copy_from_host(nullptr, 0).is_ok());
    CHECK(zero.copy_to_host(nullptr, 0).is_ok());
    CHECK(!DeviceBuffer::allocate(sizeof(int), nullptr).is_ok());

    CudaStream invalid_stream;
    CudaEvent invalid_event;
    float invalid_elapsed = 0.0F;
    CHECK(!CudaStream::create(nullptr).is_ok());
    CHECK(!invalid_stream.synchronize().is_ok());
    CHECK(!CudaEvent::create(nullptr).is_ok());
    CHECK(!invalid_event.record(invalid_stream).is_ok());
    CHECK(!invalid_event.synchronize().is_ok());
    CHECK(!CudaEvent::elapsed_time_ms(
        invalid_event, invalid_event, &invalid_elapsed).is_ok());
    CHECK(!CudaEvent::elapsed_time_ms(
        invalid_event, invalid_event, nullptr).is_ok());
    CHECK(!PinnedHostBuffer::allocate(sizeof(int), nullptr).is_ok());

    std::array<int, 5> host{1, 2, 3, 4, 5};
    DeviceBuffer buffer;
    CHECK(DeviceBuffer::allocate(sizeof(host), &buffer).is_ok());
    CHECK(buffer.size_bytes() == sizeof(host));
    CHECK(buffer.copy_from_host(host.data(), sizeof(host)).is_ok());
    CHECK(!buffer.copy_from_host(host.data(), sizeof(host), 1).is_ok());
    CHECK(!buffer.copy_from_host(
        host.data(), 1, std::numeric_limits<std::size_t>::max()).is_ok());
    CHECK(!buffer.copy_to_host(nullptr, sizeof(int)).is_ok());

    add_one<<<0, 1>>>(static_cast<int*>(buffer.data()), host.size());
    const CudaStatus invalid_launch =
        cpp_llm_engine::cuda::check_last_cuda_error("invalid add_one launch");
    CHECK(!invalid_launch.is_ok());
    CHECK(invalid_launch.operation() == "invalid add_one launch");

    add_one<<<1, 32>>>(static_cast<int*>(buffer.data()), host.size());
    CHECK(cpp_llm_engine::cuda::check_last_cuda_error("add_one launch").is_ok());
    CHECK(CudaStatus::from(cudaDeviceSynchronize(), "add_one sync").is_ok());

    std::array<int, 5> result{};
    CHECK(buffer.copy_to_host(result.data(), sizeof(result)).is_ok());
    CHECK((result == std::array<int, 5>{2, 3, 4, 5, 6}));

    DeviceBuffer moved(std::move(buffer));
    CHECK(buffer.empty());
    CHECK(buffer.data() == nullptr);
    CHECK(moved.size_bytes() == sizeof(host));
    CHECK(moved.reset().is_ok());
    CHECK(moved.empty());
    CHECK(moved.reset().is_ok());

    constexpr std::size_t async_size = 4099;
    constexpr std::size_t async_bytes = async_size * sizeof(int);
    PinnedHostBuffer pinned_input;
    PinnedHostBuffer pinned_output;
    DeviceBuffer async_buffer;
    CudaStream stream;
    CudaEvent start;
    CudaEvent end;

    CHECK(PinnedHostBuffer::allocate(0, &pinned_input).is_ok());
    CHECK(pinned_input.empty());
    CHECK(PinnedHostBuffer::allocate(async_bytes, &pinned_input).is_ok());
    CHECK(PinnedHostBuffer::allocate(async_bytes, &pinned_output).is_ok());
    CHECK(DeviceBuffer::allocate(async_bytes, &async_buffer).is_ok());
    CHECK(CudaStream::create(&stream).is_ok());
    CHECK(CudaEvent::create(&start).is_ok());
    CHECK(CudaEvent::create(&end).is_ok());

    auto* input = static_cast<int*>(pinned_input.data());
    auto* output = static_cast<int*>(pinned_output.data());
    for (std::size_t index = 0; index < async_size; ++index) {
        input[index] = static_cast<int>(index);
        output[index] = 0;
    }

    CHECK(!async_buffer.copy_from_host_async(
        input, async_bytes, invalid_stream).is_ok());
    CHECK(!async_buffer.copy_to_host_async(
        output, async_bytes, invalid_stream).is_ok());
    CHECK(async_buffer.copy_from_host_async(nullptr, 0, invalid_stream).is_ok());
    CHECK(start.record(stream).is_ok());
    CHECK(async_buffer.copy_from_host_async(input, async_bytes, stream).is_ok());

    constexpr int block_size = 256;
    const int grid_size =
        static_cast<int>((async_size + block_size - 1) / block_size);
    add_one<<<grid_size, block_size, 0, stream.native_handle()>>>(
        static_cast<int*>(async_buffer.data()), async_size);
    CHECK(cpp_llm_engine::cuda::check_last_cuda_error(
        "async add_one launch").is_ok());
    CHECK(async_buffer.copy_to_host_async(output, async_bytes, stream).is_ok());
    CHECK(end.record(stream).is_ok());
    CHECK(end.synchronize().is_ok());

    float elapsed_ms = -1.0F;
    CHECK(CudaEvent::elapsed_time_ms(start, end, &elapsed_ms).is_ok());
    CHECK(elapsed_ms >= 0.0F);
    for (std::size_t index = 0; index < async_size; ++index) {
        CHECK(output[index] == static_cast<int>(index + 1));
    }
    std::cout << "async_elapsed_ms=" << elapsed_ms << '\n';

    CudaStream moved_stream(std::move(stream));
    CudaEvent moved_end(std::move(end));
    PinnedHostBuffer moved_output(std::move(pinned_output));
    CHECK(!stream.valid());
    CHECK(!end.valid());
    CHECK(pinned_output.empty());
    CHECK(moved_stream.valid());
    CHECK(moved_end.valid());
    CHECK(moved_output.size_bytes() == async_bytes);
    CHECK(moved_stream.synchronize().is_ok());
    CHECK(moved_end.reset().is_ok());
    CHECK(moved_end.reset().is_ok());
    CHECK(moved_stream.reset().is_ok());
    CHECK(moved_stream.reset().is_ok());
    CHECK(moved_output.reset().is_ok());
    CHECK(pinned_input.reset().is_ok());
    CHECK(async_buffer.reset().is_ok());
    return 0;
}

#undef CHECK
