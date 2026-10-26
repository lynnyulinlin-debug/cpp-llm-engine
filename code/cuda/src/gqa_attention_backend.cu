// CUDA Backend：实现 score、稳定 softmax 和 value aggregation 三阶段 GQA。
// 契约：每个 query head 映射到 query_head / (Hq/Hkv) 的 KV head。
// 验证：cpp_llm_engine_cuda_gqa_attention_test 与独立 CPU 公式逐元素对照。
#include "cpp_llm_engine/cuda/gqa_attention_backend.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

constexpr std::size_t kBlockSize = 256;

__global__ void gqa_score_kernel(
    const float* query,
    const float* key,
    float* scores,
    std::size_t score_count,
    std::size_t key_tokens,
    std::size_t query_head_count,
    std::size_t kv_head_count,
    std::size_t head_dimension,
    std::size_t query_position_offset,
    bool causal,
    float scale) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= score_count) return;
    const std::size_t key_token = index % key_tokens;
    const std::size_t query_head_token = index / key_tokens;
    const std::size_t query_head = query_head_token % query_head_count;
    const std::size_t query_token = query_head_token / query_head_count;
    if (causal && key_token > query_position_offset + query_token) {
        scores[index] = -CUDART_INF_F;
        return;
    }
    const std::size_t group_size = query_head_count / kv_head_count;
    const std::size_t kv_head = query_head / group_size;
    const float* query_vector = query +
        (query_token * query_head_count + query_head) * head_dimension;
    const float* key_vector = key +
        (key_token * kv_head_count + kv_head) * head_dimension;
    float score = 0.0F;
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
        score += query_vector[dimension] * key_vector[dimension];
    }
    scores[index] = score * scale;
}

__global__ void stable_softmax_kernel(
    float* scores,
    std::size_t rows,
    std::size_t key_tokens) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float* values = scores + row * key_tokens;
    float maximum = -CUDART_INF_F;
    for (std::size_t key_token = 0; key_token < key_tokens; ++key_token) {
        maximum = fmaxf(maximum, values[key_token]);
    }
    float denominator = 0.0F;
    for (std::size_t key_token = 0; key_token < key_tokens; ++key_token) {
        const float probability = isinf(values[key_token]) && values[key_token] < 0.0F
                                      ? 0.0F
                                      : expf(values[key_token] - maximum);
        values[key_token] = probability;
        denominator += probability;
    }
    for (std::size_t key_token = 0; key_token < key_tokens; ++key_token) {
        values[key_token] /= denominator;
    }
}

__global__ void gqa_value_kernel(
    const float* probabilities,
    const float* value,
    float* output,
    std::size_t output_count,
    std::size_t key_tokens,
    std::size_t query_head_count,
    std::size_t kv_head_count,
    std::size_t head_dimension) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    const std::size_t dimension = index % head_dimension;
    const std::size_t query_head_token = index / head_dimension;
    const std::size_t query_head = query_head_token % query_head_count;
    const std::size_t group_size = query_head_count / kv_head_count;
    const std::size_t kv_head = query_head / group_size;
    const float* row = probabilities + query_head_token * key_tokens;
    float result = 0.0F;
    for (std::size_t key_token = 0; key_token < key_tokens; ++key_token) {
        const std::size_t value_index =
            (key_token * kv_head_count + kv_head) * head_dimension + dimension;
        result += row[key_token] * value[value_index];
    }
    output[index] = result;
}

CudaStatus launch_status(const char* operation) {
    return check_last_cuda_error(operation);
}

}  // namespace

core::BackendKind CudaGqaAttentionBackend::kind() const {
    return core::BackendKind::CUDA;
}

std::string_view CudaGqaAttentionBackend::name() const {
    return "cuda-gqa-attention";
}

CudaStatus CudaGqaAttentionBackend::compute(
    const core::GqaAttentionRequest& request,
    float* score_workspace,
    std::size_t workspace_elements,
    const CudaStream& stream) const {
    if (!request.validate().is_ok() || score_workspace == nullptr ||
        workspace_elements < request.score_count()) {
        return CudaStatus::from(cudaErrorInvalidValue, "CUDA GQA request/workspace");
    }
    if (!stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidResourceHandle, "CUDA GQA stream");
    }
    const std::size_t score_count = request.score_count();
    const std::size_t row_count = request.query_tokens * request.query_head_count;
    const std::size_t output_count = row_count * request.head_dimension;
    const auto blocks_for = [](std::size_t count) {
        return count / kBlockSize + (count % kBlockSize != 0 ? 1 : 0);
    };
    const std::size_t score_blocks = blocks_for(score_count);
    const std::size_t row_blocks = blocks_for(row_count);
    const std::size_t output_blocks = blocks_for(output_count);
    if (score_blocks > std::numeric_limits<unsigned int>::max() ||
        row_blocks > std::numeric_limits<unsigned int>::max() ||
        output_blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, "CUDA GQA grid");
    }
    const float scale = request.scale == 0.0F
                            ? 1.0F / sqrtf(static_cast<float>(request.head_dimension))
                            : request.scale;
    gqa_score_kernel<<<static_cast<unsigned int>(score_blocks), kBlockSize, 0,
                       stream.native_handle()>>>(
        request.query, request.key, score_workspace, score_count,
        request.key_tokens, request.query_head_count, request.kv_head_count,
        request.head_dimension, request.query_position_offset, request.causal, scale);
    CudaStatus status = launch_status("gqa_score_kernel launch");
    if (!status.is_ok()) return status;
    stable_softmax_kernel<<<static_cast<unsigned int>(row_blocks), kBlockSize, 0,
                            stream.native_handle()>>>(
        score_workspace, row_count, request.key_tokens);
    status = launch_status("stable_softmax_kernel launch");
    if (!status.is_ok()) return status;
    gqa_value_kernel<<<static_cast<unsigned int>(output_blocks), kBlockSize, 0,
                       stream.native_handle()>>>(
        score_workspace, request.value, request.output, output_count,
        request.key_tokens, request.query_head_count, request.kv_head_count,
        request.head_dimension);
    return launch_status("gqa_value_kernel launch");
}

const CudaGqaAttentionBackend& cuda_gqa_attention_backend() {
    static const CudaGqaAttentionBackend backend;
    return backend;
}

}  // namespace cpp_llm_engine::cuda
