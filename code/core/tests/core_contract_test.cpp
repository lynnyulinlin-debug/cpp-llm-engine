// Core：公共契约测试
// 职责：验证 Backend、Operator、Tensor、Decoder 元数据和错误状态的最小行为。
// 通过条件：所有 assert 成立，且目标由 core CTest 执行。
// 验证：由 Chapter 03/04/06 或 scripts/run_regression_baseline.sh 间接构建执行。
#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/core/decoder_contract.h"
#include "cpp_llm_engine/core/gqa_attention_contract.h"
#include "cpp_llm_engine/core/operator.h"
#include "cpp_llm_engine/core/quantization_contract.h"
#include "cpp_llm_engine/core/rope_contract.h"
#include "cpp_llm_engine/core/tensor_contract.h"

#include <cassert>
#include <limits>
#include <string_view>

namespace {

class CpuBackend final : public cpp_llm_engine::core::Backend {
public:
    cpp_llm_engine::core::BackendKind kind() const override {
        return cpp_llm_engine::core::BackendKind::CPU;
    }
    std::string_view name() const override { return "cpu-reference"; }
};

class MatmulOperator final : public cpp_llm_engine::core::Operator {
public:
    std::string_view name() const override { return "matmul"; }
};

}  // namespace

int main() {
    using namespace cpp_llm_engine::core;

    Shape shape{{2, 3, 4}};
    std::size_t elements = 0;
    assert(shape.checked_numel(&elements).is_ok());
    assert(elements == 24);

    Shape overflow{{std::numeric_limits<std::size_t>::max(), 2}};
    assert(!overflow.checked_numel(&elements).is_ok());

    const TensorContract tensor{
        shape, {12, 4, 1}, DType::Float32, DeviceKind::CPU, 0,
        24 * sizeof(float)};
    assert(tensor.device == DeviceKind::CPU);
    assert(tensor.validate_element_view(sizeof(float)).is_ok());

    const TensorContract offset_view{
        Shape{{2, 2}}, {3, 1}, DType::Float32, DeviceKind::CPU, 1,
        6 * sizeof(float)};
    std::size_t required_elements = 0;
    assert(offset_view.checked_required_elements(&required_elements).is_ok());
    assert(required_elements == 6);
    assert(offset_view.validate_element_view(sizeof(float)).is_ok());

    const TensorContract short_storage{
        Shape{{2, 2}}, {3, 1}, DType::Float32, DeviceKind::CPU, 1,
        5 * sizeof(float)};
    assert(!short_storage.validate_element_view(sizeof(float)).is_ok());
    const TensorContract bad_rank{
        Shape{{2, 2}}, {2}, DType::Float32, DeviceKind::CPU, 0,
        4 * sizeof(float)};
    assert(!bad_rank.validate_element_view(sizeof(float)).is_ok());
    assert(static_cast<int>(DType::Q6K) > static_cast<int>(DType::Q4K));
    std::size_t quantized_bytes = 0;
    assert(checked_quantized_matrix_bytes(
        WeightFormat::Q4K, 2, 512, &quantized_bytes).is_ok());
    assert(quantized_bytes == 4 * kQ4KBlockBytes);
    assert(checked_quantized_matrix_bytes(
        WeightFormat::Q6K, 3, 256, &quantized_bytes).is_ok());
    assert(quantized_bytes == 3 * kQ6KBlockBytes);
    assert(checked_quantized_matrix_bytes(
        WeightFormat::Q8_0, 2, 96, &quantized_bytes).is_ok());
    assert(quantized_bytes == 6 * kQ8_0BlockBytes);
    assert(!checked_quantized_matrix_bytes(
        WeightFormat::Float32, 1, 256, &quantized_bytes).is_ok());
    assert(!checked_quantized_matrix_bytes(
        WeightFormat::Q4K, 1, 255, &quantized_bytes).is_ok());
    assert(!checked_quantized_matrix_bytes(
        WeightFormat::Q4K,
        std::numeric_limits<std::size_t>::max(),
        256,
        &quantized_bytes).is_ok());
    assert(Status::ok().is_ok());
    assert(!Status::error("invalid shape").is_ok());

    const DecoderLayerContract decoder_layer{
        2048, 5632, 32, 4, 64, 64,
        DecoderNormalizationKind::RmsNorm,
        DecoderActivationKind::SwiGlu,
        DecoderPositionEncodingKind::Rotary,
        true, true};
    assert(decoder_layer.validate().is_ok());
    const DecoderModelContract decoder_model{32000, 22, 2048, decoder_layer};
    assert(decoder_model.validate().is_ok());
    const DecoderEmbeddingSequence embeddings{2, 2048, std::vector<float>(4096, 0.0f)};
    assert(embeddings.validate(2048).is_ok());
    assert(!embeddings.validate(960).is_ok());
    auto invalid_embeddings = embeddings;
    invalid_embeddings.values.pop_back();
    assert(!invalid_embeddings.validate(2048).is_ok());
    auto invalid_heads = decoder_layer;
    invalid_heads.kv_head_count = 3;
    assert(!invalid_heads.validate().is_ok());
    auto invalid_rope = decoder_layer;
    invalid_rope.position_dimension = 65;
    assert(!invalid_rope.validate().is_ok());
    float rope_values[16]{};
    const RotaryEmbeddingRequest rope_request{
        rope_values, 1, 2, 8, 8, 0, 10000.0F};
    assert(rope_request.validate().is_ok());
    auto odd_rope = rope_request;
    odd_rope.rotary_dimension = 7;
    assert(!odd_rope.validate().is_ok());
    auto overflow_rope = rope_request;
    overflow_rope.token_count = 2;
    overflow_rope.position_offset = std::numeric_limits<std::size_t>::max();
    assert(!overflow_rope.validate().is_ok());
    float query[16]{};
    float key[8]{};
    float value[8]{};
    float attention_output[16]{};
    const GqaAttentionRequest gqa_request{
        query, key, value, attention_output, 1, 1, 4, 2, 4, 0, true, 0.0F};
    assert(gqa_request.validate().is_ok());
    assert(gqa_request.score_count() == 4);
    auto invalid_gqa_heads = gqa_request;
    invalid_gqa_heads.query_head_count = 3;
    assert(!invalid_gqa_heads.validate().is_ok());
    auto invalid_gqa_scale = gqa_request;
    invalid_gqa_scale.scale = -1.0F;
    assert(!invalid_gqa_scale.validate().is_ok());

    CpuBackend backend;
    MatmulOperator matmul;
    assert(backend.kind() == BackendKind::CPU);
    assert(backend.name() == "cpu-reference");
    assert(matmul.name() == "matmul");
    return 0;
}
