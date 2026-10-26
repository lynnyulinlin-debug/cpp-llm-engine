// CUDA Decoder：验证合成单 token Decoder 层的完整顺序和中间 Tensor。
// 职责：用可读小维度权重检查 RMSNorm、GQA、Residual 与 SwiGLU 组装。
// 通过条件：关键中间值和最终 hidden 与独立 CPU 计算一致，KV 增加一个 token。
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_decoder_layer.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

namespace {
using cpp_llm_engine::core::DType;
using namespace cpp_llm_engine::cuda;

std::vector<float> rms(const std::vector<float>& x) {
    float sum = 0.0F;
    for (float v : x) sum += v * v;
    const float inv = 1.0F / std::sqrt(sum / x.size() + 1.0e-5F);
    std::vector<float> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv;
    return y;
}

bool close(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 4.0e-5F) return false;
    }
    return true;
}

}  // namespace

int main() {
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);
    CudaExecutionContext context;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    const CudaDecoderLayerConfig config{8, 12, 4, 2, 2, 2, 1.0e-5F, 10000.0F};
    CudaDecoderLayer layer;
    CudaKvCache cache;
    CHECK(CudaDecoderLayer::create(config, &layer).is_ok());
    CHECK(CudaKvCache::create(4, 2, 2, &cache).is_ok());

    std::vector<CudaTensor> tensors(9);
    const std::vector<std::vector<std::size_t>> shapes{
        {8}, {8,8}, {8,4}, {8,4}, {8,8}, {8}, {8,12}, {8,12}, {12,8}};
    std::vector<std::vector<float>> host;
    for (const auto& shape : shapes) {
        std::size_t count = 1;
        for (auto dim : shape) count *= dim;
        host.emplace_back(count, 0.0F);
    }
    host[0].assign(8, 1.0F);
    host[5].assign(8, 1.0F);
    for (std::size_t i = 0; i < 8; ++i) {
        host[1][i * 8 + i] = 1.0F;
        host[4][i * 8 + i] = 1.0F;
        host[6][i * 12 + i] = 0.5F;
        host[7][i * 12 + i] = 0.25F;
        host[8][i * 8 + i] = 0.75F;
    }
    for (std::size_t i = 0; i < 4; ++i) {
        host[2][i * 4 + i] = 1.0F;
        host[3][i * 4 + i] = 1.0F;
    }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        CHECK(CudaTensor::create_contiguous(shapes[i], DType::Float32, &tensors[i]).is_ok());
        CHECK(context.upload(&tensors[i], host[i].data(), host[i].size() * sizeof(float)).is_ok());
    }
    const CudaDecoderLayerWeights weights{
        &tensors[0], CudaLinearWeight::float32(&tensors[1]),
        CudaLinearWeight::float32(&tensors[2]), CudaLinearWeight::float32(&tensors[3]),
        CudaLinearWeight::float32(&tensors[4]), &tensors[5],
        CudaLinearWeight::float32(&tensors[6]), CudaLinearWeight::float32(&tensors[7]),
        CudaLinearWeight::float32(&tensors[8])};
    std::vector<float> hidden_host{0.2F,-0.4F,0.6F,-0.8F,1.0F,-1.2F,1.4F,-1.6F};
    CudaTensor hidden;
    CHECK(CudaTensor::create_contiguous({8}, DType::Float32, &hidden).is_ok());
    CHECK(context.upload(&hidden, hidden_host.data(), 8 * sizeof(float)).is_ok());
    CHECK(layer.forward(&context, weights, &cache, &hidden, 0).is_ok());

    const std::vector<float> norm1 = rms(hidden_host);
    std::vector<float> attention{norm1[0], norm1[1], norm1[0], norm1[1],
                                 norm1[2], norm1[3], norm1[2], norm1[3]};
    std::vector<float> after_attention(8);
    for (std::size_t i = 0; i < 8; ++i) after_attention[i] = hidden_host[i] + attention[i];
    const std::vector<float> norm2 = rms(after_attention);
    std::vector<float> swiglu(12, 0.0F);
    for (std::size_t i = 0; i < 8; ++i) {
        const float gate = norm2[i] * 0.5F;
        swiglu[i] = gate / (1.0F + std::exp(-gate)) * (norm2[i] * 0.25F);
    }
    std::vector<float> expected(8);
    for (std::size_t i = 0; i < 8; ++i) expected[i] = after_attention[i] + swiglu[i] * 0.75F;
    const auto check_tensor = [&](const CudaTensor& tensor, const std::vector<float>& golden) {
        std::vector<float> actual(golden.size());
        return context.download(tensor, actual.data(), actual.size() * sizeof(float)).is_ok() &&
               context.synchronize().is_ok() && close(actual, golden);
    };
    CHECK(check_tensor(layer.attention_normalized(), norm1));
    CHECK(check_tensor(layer.attention_context(), attention));
    CHECK(check_tensor(layer.ffn_normalized(), norm2));
    CHECK(check_tensor(layer.swiglu_output(), swiglu));
    CHECK(check_tensor(hidden, expected));
    CHECK(cache.size_tokens() == 1);
    CHECK(!layer.forward(nullptr, weights, &cache, &hidden, 1).is_ok());
    return 0;
}

#undef CHECK
