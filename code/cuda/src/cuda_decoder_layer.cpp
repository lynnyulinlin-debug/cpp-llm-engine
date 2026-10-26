// CUDA Decoder：实现单 token Llama Decoder 层的可解释组装。
// 职责：复用 ExecutionContext，不在层内实现数学 kernel 或拥有请求 KV。
// 契约：forward 成功后 hidden 原地更新，cache 增加一个 token。
// 验证：cpp_llm_engine_cuda_decoder_layer_test。
#include "cpp_llm_engine/cuda/cuda_decoder_layer.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"

#include <cmath>
#include <utility>
#include <vector>

namespace cpp_llm_engine::cuda {

core::Status CudaDecoderLayer::create(
    const CudaDecoderLayerConfig& config, CudaDecoderLayer* output) {
    if (output == nullptr || config.hidden_size == 0 || config.ffn_size == 0 ||
        config.query_heads == 0 || config.kv_heads == 0 ||
        config.query_heads % config.kv_heads != 0 || config.head_dimension == 0 ||
        config.query_heads * config.head_dimension != config.hidden_size ||
        config.rotary_dimension == 0 || config.rotary_dimension > config.head_dimension ||
        config.rotary_dimension % 2 != 0 || config.rms_epsilon <= 0.0F ||
        !std::isfinite(config.rms_epsilon) || config.rope_frequency_base <= 0.0F) {
        return core::Status::error("CudaDecoderLayer config is invalid");
    }
    CudaDecoderLayer replacement;
    replacement.config_ = config;
    const std::vector<std::vector<std::size_t>> shapes{
        {config.hidden_size}, {1, config.query_heads, config.head_dimension},
        {1, config.kv_heads, config.head_dimension}, {1, config.kv_heads, config.head_dimension},
        {1, config.query_heads, config.head_dimension}, {config.hidden_size},
        {config.hidden_size}, {config.ffn_size}, {config.ffn_size},
        {config.ffn_size}, {config.hidden_size}};
    CudaTensor* tensors[]{
        &replacement.attention_normalized_, &replacement.query_, &replacement.key_,
        &replacement.value_, &replacement.attention_context_, &replacement.projection_,
        &replacement.ffn_normalized_, &replacement.gate_, &replacement.up_,
        &replacement.swiglu_, &replacement.down_};
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        const core::Status status = CudaTensor::create_contiguous(
            shapes[index], core::DType::Float32, tensors[index]);
        if (!status.is_ok()) return status;
    }
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaDecoderLayer::forward(
    CudaExecutionContext* context, const CudaDecoderLayerWeights& weights,
    CudaKvCache* cache, CudaTensor* hidden, std::size_t position) {
    if (context == nullptr || cache == nullptr || hidden == nullptr) {
        return core::Status::error("CudaDecoderLayer forward input is null");
    }
    if (weights.attention_norm == nullptr || weights.ffn_norm == nullptr ||
        !weights.query.valid() || !weights.key.valid() || !weights.value.valid() ||
        !weights.attention_output.valid() || !weights.gate.valid() ||
        !weights.up.valid() || !weights.down.valid()) {
        return core::Status::error("CudaDecoderLayer weight is invalid");
    }
#define RETURN_IF_ERROR(expression) do { const auto status = (expression); \
    if (!status.is_ok()) return status; } while (false)
    RETURN_IF_ERROR(context->rms_norm(
        *hidden, *weights.attention_norm, &attention_normalized_, config_.rms_epsilon));
    RETURN_IF_ERROR(context->linear(attention_normalized_, weights.query, &query_));
    RETURN_IF_ERROR(context->linear(attention_normalized_, weights.key, &key_));
    RETURN_IF_ERROR(context->linear(attention_normalized_, weights.value, &value_));
    RETURN_IF_ERROR(context->rope(
        &query_, position, config_.rotary_dimension, config_.rope_frequency_base));
    RETURN_IF_ERROR(context->rope(
        &key_, position, config_.rotary_dimension, config_.rope_frequency_base));
    RETURN_IF_ERROR(context->append_kv(cache, key_, value_));
    RETURN_IF_ERROR(context->gqa_attention(
        query_, *cache, &attention_context_, position));
    RETURN_IF_ERROR(context->linear(
        attention_context_, weights.attention_output, &projection_));
    RETURN_IF_ERROR(context->vector_add(*hidden, projection_, hidden));
    RETURN_IF_ERROR(context->rms_norm(
        *hidden, *weights.ffn_norm, &ffn_normalized_, config_.rms_epsilon));
    RETURN_IF_ERROR(context->linear(ffn_normalized_, weights.gate, &gate_));
    RETURN_IF_ERROR(context->linear(ffn_normalized_, weights.up, &up_));
    RETURN_IF_ERROR(context->swiglu(gate_, up_, &swiglu_));
    RETURN_IF_ERROR(context->linear(swiglu_, weights.down, &down_));
    RETURN_IF_ERROR(context->vector_add(*hidden, down_, hidden));
#undef RETURN_IF_ERROR
    return core::Status::ok();
}

const CudaTensor& CudaDecoderLayer::attention_normalized() const { return attention_normalized_; }
const CudaTensor& CudaDecoderLayer::attention_context() const { return attention_context_; }
const CudaTensor& CudaDecoderLayer::ffn_normalized() const { return ffn_normalized_; }
const CudaTensor& CudaDecoderLayer::swiglu_output() const { return swiglu_; }

}  // namespace cpp_llm_engine::cuda
