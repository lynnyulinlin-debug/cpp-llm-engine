// CUDA Decoder：验证 RMSNorm 与 SwiGLU 的数值和错误路径。
// 职责：使用 257/2048 宽度 CPU golden 覆盖非整除 grid 与 TinyLlama hidden size。
// 通过条件：逐元素误差通过，shape、epsilon 和空输出错误被拒绝。
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
    std::cerr << "check failed: " #condition " at line " << __LINE__ << '\n'; \
    return false; } } while (false)

namespace {

using cpp_llm_engine::core::DType;
using cpp_llm_engine::cuda::CudaExecutionContext;
using cpp_llm_engine::cuda::CudaTensor;

bool run_case(std::size_t elements) {
    CudaExecutionContext context;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    std::vector<float> input(elements);
    std::vector<float> weight(elements);
    std::vector<float> up(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 19) - 9) * 0.07F;
        weight[index] = 0.8F + static_cast<float>(index % 5) * 0.05F;
        up[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.04F;
    }
    CudaTensor input_tensor;
    CudaTensor weight_tensor;
    CudaTensor up_tensor;
    CudaTensor norm_output;
    CudaTensor swiglu_output;
    for (CudaTensor* tensor : {&input_tensor, &weight_tensor, &up_tensor,
                               &norm_output, &swiglu_output}) {
        CHECK(CudaTensor::create_contiguous({elements}, DType::Float32, tensor).is_ok());
    }
    CHECK(context.upload(&input_tensor, input.data(), elements * sizeof(float)).is_ok());
    CHECK(context.upload(&weight_tensor, weight.data(), elements * sizeof(float)).is_ok());
    CHECK(context.upload(&up_tensor, up.data(), elements * sizeof(float)).is_ok());
    constexpr float epsilon = 1.0e-5F;
    CHECK(context.rms_norm(input_tensor, weight_tensor, &norm_output, epsilon).is_ok());
    CHECK(context.swiglu(input_tensor, up_tensor, &swiglu_output).is_ok());
    std::vector<float> norm(elements);
    std::vector<float> swiglu(elements);
    CHECK(context.download(norm_output, norm.data(), elements * sizeof(float)).is_ok());
    CHECK(context.download(swiglu_output, swiglu.data(), elements * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    float mean_square = 0.0F;
    for (const float value : input) mean_square += value * value;
    const float inverse = 1.0F /
        std::sqrt(mean_square / static_cast<float>(elements) + epsilon);
    for (std::size_t index = 0; index < elements; ++index) {
        const float expected_norm = input[index] * inverse * weight[index];
        const float expected_swiglu =
            input[index] / (1.0F + std::exp(-input[index])) * up[index];
        CHECK(std::fabs(norm[index] - expected_norm) <= 2.0e-5F);
        CHECK(std::fabs(swiglu[index] - expected_swiglu) <= 2.0e-6F);
    }
    CudaTensor wrong;
    CHECK(CudaTensor::create_contiguous({elements + 1}, DType::Float32, &wrong).is_ok());
    CHECK(!context.rms_norm(input_tensor, wrong, &norm_output, epsilon).is_ok());
    CHECK(!context.rms_norm(input_tensor, weight_tensor, &norm_output, 0.0F).is_ok());
    CHECK(!context.swiglu(input_tensor, up_tensor, nullptr).is_ok());
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) return 1;
    if (!run_case(257)) return 1;
    if (!run_case(2048)) return 1;
    return 0;
}

#undef CHECK
