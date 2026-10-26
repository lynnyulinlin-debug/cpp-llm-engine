// Engine CUDA：验证 Chapter01 layer 权重到 CUDA Model 权重的上传边界。
// 职责：检查 F32 norm、Q4_K/Q6_K 格式、矩阵方向、payload bytes 和失败路径。
// 通过条件：九个权重上传后视图有效，host 销毁不影响 device 所有权。
#include "cpp_llm_engine/cuda_llama_layer_weights.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "real_layer_weights.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

namespace {

chapter01::Tensor packed(
    const std::string& name, chapter01::DType dtype, std::size_t rows) {
    const std::size_t block_bytes = dtype == chapter01::DType::Q4_K ? 144 : 210;
    std::vector<std::byte> bytes(rows * block_bytes, std::byte{0});
    return chapter01::Tensor::from_raw_bytes(
        name, {256, static_cast<std::int64_t>(rows)}, dtype, std::move(bytes));
}

chapter01::RealLlamaLayerWeights make_layer() {
    chapter01::RealLlamaLayerWeights layer;
    layer.attn_norm = chapter01::Tensor::from_vector<float>(
        "attn_norm", {256}, std::vector<float>(256, 1.0F));
    layer.ffn_norm = chapter01::Tensor::from_vector<float>(
        "ffn_norm", {256}, std::vector<float>(256, 1.0F));
    layer.attn_q = packed("q", chapter01::DType::Q4_K, 256);
    layer.attn_k = packed("k", chapter01::DType::Q4_K, 64);
    layer.attn_v = packed("v", chapter01::DType::Q4_K, 64);
    layer.attn_output = packed("o", chapter01::DType::Q4_K, 256);
    layer.ffn_gate = packed("gate", chapter01::DType::Q4_K, 512);
    layer.ffn_up = packed("up", chapter01::DType::Q4_K, 512);
    layer.ffn_down = packed("down", chapter01::DType::Q6_K, 256);
    return layer;
}

}  // namespace

int main() {
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);
    cpp_llm_engine::cuda::CudaExecutionContext context;
    CHECK(cpp_llm_engine::cuda::CudaExecutionContext::create(&context).is_ok());
    chapter01::RealLlamaLayerWeights host = make_layer();
    cpp_llm_engine::CudaLlamaLayerWeights device;
    CHECK(cpp_llm_engine::CudaLlamaLayerWeights::upload(
        &context, host, &device).is_ok());
    CHECK(device.valid());
    CHECK(device.query().rows() == 256);
    CHECK(device.query().cols() == 256);
    CHECK(device.query().size_bytes() == 256 * 144);
    CHECK(device.query().format() == cpp_llm_engine::core::WeightFormat::Q4K);
    CHECK(device.down().rows() == 256);
    CHECK(device.down().cols() == 256);
    CHECK(device.down().size_bytes() == 256 * 210);
    CHECK(device.down().format() == cpp_llm_engine::core::WeightFormat::Q6K);
    host = {};
    CHECK(device.valid());
    CHECK(device.decoder_weights().key.valid());

    chapter01::RealLlamaLayerWeights invalid = make_layer();
    invalid.attn_norm.set_dtype(chapter01::DType::Float16);
    cpp_llm_engine::CudaLlamaLayerWeights rejected;
    CHECK(!cpp_llm_engine::CudaLlamaLayerWeights::upload(
        &context, invalid, &rejected).is_ok());
    CHECK(!rejected.valid());
    CHECK(!cpp_llm_engine::CudaLlamaLayerWeights::upload(
        nullptr, invalid, &rejected).is_ok());
    return 0;
}

#undef CHECK
