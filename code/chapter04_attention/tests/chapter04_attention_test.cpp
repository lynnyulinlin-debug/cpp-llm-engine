// Chapter 04：Attention/Decoder 综合测试
// 职责：验证 causal mask、decode offset、KV Cache、LayerNorm、FFN 和 DecoderLayer。
// 通过条件：所有 assert 成立，Chapter 04 与 core CTest 均通过。
// 验证：bash code/chapter04_attention/scripts/run.sh。
#include "chapter04_attention/decoder_layer.h"
#include "chapter04_attention/attention_backend.h"
#include "chapter04_attention/kv_cache.h"

#include <cassert>
#include <cmath>

namespace {

chapter04_attention::Tensor make_input() {
    chapter04_attention::Tensor input(2, 4);
    input.at(0, 0) = 1.0f;
    input.at(0, 1) = 2.0f;
    input.at(0, 2) = 3.0f;
    input.at(0, 3) = 4.0f;
    input.at(1, 0) = 5.0f;
    input.at(1, 1) = 6.0f;
    input.at(1, 2) = 7.0f;
    input.at(1, 3) = 8.0f;
    return input;
}

}  // namespace

int main() {
    const float query[] = {1.0f, 1.0f};
    const float key[] = {1.0f, 1.0f};
    const float value[] = {2.0f, 4.0f};
    float attention_output[] = {0.0f, 0.0f};
    const cpp_llm_engine::core::AttentionRequest causal_request{
        query, key, value, attention_output, 2, 1, 1, 1, 0, true, 1.0f};
    assert(chapter04_attention::cpu_reference_attention_backend().compute(causal_request).is_ok());
    assert(std::abs(attention_output[0] - 2.0f) < 1e-5f);

    const float two_keys[] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float two_values[] = {2.0f, 4.0f, 100.0f, 200.0f};
    float first_decode_output[] = {0.0f, 0.0f};
    const cpp_llm_engine::core::AttentionRequest future_mask_request{
        query, two_keys, two_values, first_decode_output, 1, 2, 1, 2, 0, true, 1.0f};
    assert(chapter04_attention::cpu_reference_attention_backend().compute(
               future_mask_request).is_ok());
    assert(std::abs(first_decode_output[0] - 2.0f) < 1e-5f);

    chapter04_attention::KvCache cache(2, 1, 2);
    assert(cache.ready());
    assert(cache.width() == 2);
    assert(cache.append(two_keys, two_values, 2));
    assert(cache.size() == 2);
    assert(cache.key_data()[0] == 1.0f);
    assert(cache.value_data()[2] == 100.0f);
    assert(!cache.append(two_keys, two_values, 1));

    float cached_output[] = {0.0f, 0.0f};
    const cpp_llm_engine::core::AttentionRequest cached_decode_request{
        query, cache.key_data(), cache.value_data(), cached_output,
        1, cache.size(), 1, 2, 1, true, 1.0f};
    assert(chapter04_attention::cpu_reference_attention_backend().compute(
               cached_decode_request).is_ok());
    assert(std::abs(cached_output[0] - 51.0f) < 1e-5f);
    cache.clear();
    assert(cache.size() == 0);
    assert(cache.append(two_keys, two_values, 1));

    const chapter04_attention::Tensor input = make_input();

    chapter04_attention::LayerNorm norm(4);
    const chapter04_attention::Tensor normed = norm.forward(input);
    assert(norm.ready());
    assert(normed.ready());
    assert(normed.rows == 2);
    assert(normed.cols == 4);

    chapter04_attention::MultiHeadAttention attention(4, 2);
    const chapter04_attention::Tensor attended = attention.forward(input);
    assert(attention.ready());
    assert(attended.ready());
    assert(attended.rows == 2);
    assert(attended.cols == 4);
    assert(attention.backend_kind() == cpp_llm_engine::core::BackendKind::CPU);

    chapter04_attention::FeedForward feed_forward(4, 8);
    const chapter04_attention::Tensor ffn_output = feed_forward.forward(input);
    assert(feed_forward.ready());
    assert(ffn_output.ready());
    assert(ffn_output.rows == 2);
    assert(ffn_output.cols == 4);

    chapter04_attention::DecoderLayer layer(4, 2, 8, 0);
    const chapter04_attention::Tensor output = layer.forward(input);
    assert(layer.ready());
    assert(output.ready());
    assert(output.rows == 2);
    assert(output.cols == 4);
    assert(std::abs(output.at(0, 0) - input.at(0, 0)) > 1e-4f);
    assert(std::abs(output.at(1, 3) - input.at(1, 3)) > 1e-4f);
    const auto layer_contract = layer.contract();
    assert(layer_contract.validate().is_ok());
    assert(layer_contract.hidden_size == 4);
    assert(layer_contract.ffn_hidden_size == 8);
    assert(layer_contract.query_head_count == 2);
    assert(layer_contract.kv_head_count == 2);
    assert(layer_contract.normalization ==
           cpp_llm_engine::core::DecoderNormalizationKind::LayerNorm);
    assert(layer_contract.activation ==
           cpp_llm_engine::core::DecoderActivationKind::Relu);
    assert(!layer_contract.uses_kv_cache);

    return 0;
}
