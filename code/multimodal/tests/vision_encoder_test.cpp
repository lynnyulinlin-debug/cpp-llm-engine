// 多模态 CPU：SigLIP Patch Embedding 与单 block 测试
// 职责：用小型确定性权重验证 position、LayerNorm、全注意力、GELU 和 residual 顺序。
// 契约：测试不依赖真实 mmproj、Python、网络、llama.cpp 或 GPU。
// 运行：由 CTest 执行 multimodal_vision_encoder_test。
#include "multimodal/vision_encoder.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                                   \
    do {                                                                                   \
        if (!(condition)) {                                                                \
            std::cerr << "CHECK failed at line " << __LINE__ << ": " #condition << '\n'; \
            ++failures;                                                                    \
        }                                                                                  \
    } while (false)

bool near(float lhs, float rhs, float tolerance = 1.0e-5f) {
    return std::fabs(lhs - rhs) <= tolerance;
}

multimodal::LinearWeights linear(std::size_t input,
                                 std::size_t output,
                                 std::vector<float> weight,
                                 std::vector<float> bias = {}) {
    return {input, output, std::move(weight), std::move(bias)};
}

multimodal::LinearWeights identity(std::size_t size) {
    std::vector<float> weight(size * size, 0.0f);
    for (std::size_t i = 0; i < size; ++i) weight[i * size + i] = 1.0f;
    return linear(size, size, std::move(weight), std::vector<float>(size, 0.0f));
}

multimodal::LinearWeights patterned_linear(std::size_t input, std::size_t output) {
    std::vector<float> weight(input * output);
    std::vector<float> bias(output);
    for (std::size_t out = 0; out < output; ++out) {
        bias[out] = static_cast<float>(static_cast<int>(out % 3) - 1) * 0.03f;
        for (std::size_t in = 0; in < input; ++in) {
            const int pattern = static_cast<int>((out * input + in) % 7) - 3;
            weight[out * input + in] = static_cast<float>(pattern) * 0.04f;
        }
    }
    return linear(input, output, std::move(weight), std::move(bias));
}

void test_patch_and_position_embedding() {
    const multimodal::PatchTensor patches{2, 4, {1, 2, 3, 4, 5, 6, 7, 8}};
    const multimodal::PatchEmbeddingWeights weights{
        4,
        2,
        2,
        {1, 0, 0, 0,
         0, 0, 0, 1},
        {0.5f, -0.5f},
        {0.1f, 0.2f,
         0.3f, 0.4f}};

    multimodal::FloatMatrix output;
    std::string error;
    CHECK(multimodal::embed_patches(patches, weights, &output, &error));
    CHECK(error.empty());
    CHECK(output.rows == 2 && output.cols == 2);
    CHECK(near(output.at(0, 0), 1.6f));
    CHECK(near(output.at(0, 1), 3.7f));
    CHECK(near(output.at(1, 0), 5.8f));
    CHECK(near(output.at(1, 1), 7.9f));

    const std::vector<std::size_t> swapped_positions{1, 0};
    CHECK(multimodal::embed_patches_at_positions(
        patches, swapped_positions, weights, &output, &error));
    CHECK(near(output.at(0, 0), 1.8f));
    CHECK(near(output.at(0, 1), 3.9f));
    CHECK(near(output.at(1, 0), 5.6f));
    CHECK(near(output.at(1, 1), 7.7f));
    CHECK(!multimodal::embed_patches_at_positions(
        patches, {0}, weights, &output, &error));

    auto too_few_positions = weights;
    too_few_positions.position_count = 1;
    too_few_positions.position_weight.resize(2);
    CHECK(!multimodal::embed_patches(patches, too_few_positions, &output, &error));
    CHECK(error.find("position") != std::string::npos);
    CHECK(!multimodal::embed_patches(patches, weights, nullptr, &error));
}

void test_siglip_block_order() {
    const multimodal::FloatMatrix input{2, 2, {1.0f, 3.0f, 2.0f, 0.0f}};
    multimodal::SiglipBlockWeights weights;
    weights.layer_norm_1 = {{2.0f, 3.0f}, {0.5f, -0.5f}};
    weights.query = linear(2, 2, {0, 0, 0, 0}, {0, 0});
    weights.key = linear(2, 2, {0, 0, 0, 0}, {0, 0});
    weights.value = linear(2, 2, {0, 0, 0, 0}, {0.25f, -0.5f});
    weights.attention_output = identity(2);
    weights.layer_norm_2 = {{1.0f, 1.0f}, {0.0f, 0.0f}};
    weights.feed_forward_input = linear(2, 2, {0, 0, 0, 0}, {1.0f, -1.0f});
    weights.feed_forward_output = identity(2);

    multimodal::SiglipBlockTrace trace;
    std::string error;
    CHECK(multimodal::run_siglip_block(input, 1, 1.0e-5f, weights, &trace, &error));
    CHECK(error.empty());
    CHECK(near(trace.normalized_attention_input.at(0, 0), -1.49999f, 2.0e-5f));
    CHECK(near(trace.normalized_attention_input.at(0, 1), 2.49999f, 2.0e-5f));
    CHECK(near(trace.attention_output.at(0, 0), 0.25f));
    CHECK(near(trace.attention_output.at(1, 1), -0.5f));
    CHECK(near(trace.attention_residual.at(0, 0), 1.25f));
    CHECK(near(trace.attention_residual.at(1, 1), -0.5f));
    CHECK(near(trace.feed_forward_hidden.at(0, 0), 0.841192f, 1.0e-5f));
    CHECK(near(trace.feed_forward_hidden.at(0, 1), -0.158808f, 1.0e-5f));
    CHECK(near(trace.output.at(0, 0), 2.091192f, 1.0e-5f));
    CHECK(near(trace.output.at(0, 1), 2.341192f, 1.0e-5f));
    CHECK(near(trace.output.at(1, 0), 3.091192f, 1.0e-5f));
    CHECK(near(trace.output.at(1, 1), -0.658808f, 1.0e-5f));

    CHECK(!multimodal::run_siglip_block(input, 0, 1.0e-5f, weights, &trace, &error));
    CHECK(error.find("heads") != std::string::npos);
    auto invalid = weights;
    invalid.query.weight.pop_back();
    CHECK(!multimodal::run_siglip_block(input, 1, 1.0e-5f, invalid, &trace, &error));
    CHECK(error.find("weights") != std::string::npos);
    CHECK(!multimodal::run_siglip_block(input, 1, 1.0e-5f, weights, nullptr, &error));
}

void test_siglip_encoder_streaming_weights() {
    const multimodal::FloatMatrix input{2, 2, {1.0f, 3.0f, 2.0f, 0.0f}};
    multimodal::SiglipBlockWeights weights;
    weights.layer_norm_1 = {{1.0f, 1.0f}, {0.0f, 0.0f}};
    weights.query = linear(2, 2, {0, 0, 0, 0}, {0, 0});
    weights.key = linear(2, 2, {0, 0, 0, 0}, {0, 0});
    weights.value = linear(2, 2, {0, 0, 0, 0}, {0.1f, -0.2f});
    weights.attention_output = identity(2);
    weights.layer_norm_2 = {{1.0f, 1.0f}, {0.0f, 0.0f}};
    weights.feed_forward_input = linear(2, 2, {0, 0, 0, 0}, {0.5f, -0.5f});
    weights.feed_forward_output = identity(2);
    const multimodal::LayerNormWeights post_norm{{1.0f, 1.0f}, {0.0f, 0.0f}};

    multimodal::SiglipBlockTrace first;
    multimodal::SiglipBlockTrace second;
    multimodal::FloatMatrix expected;
    std::string error;
    CHECK(multimodal::run_siglip_block(input, 1, 1.0e-5f, weights, &first, &error));
    CHECK(multimodal::run_siglip_block(first.output, 1, 1.0e-5f, weights, &second, &error));
    CHECK(multimodal::apply_layer_norm(second.output, post_norm, 1.0e-5f, &expected, &error));

    std::vector<std::size_t> requested;
    const multimodal::SiglipBlockWeightProvider provider =
        [&requested, &weights](std::size_t index,
                               multimodal::SiglipBlockWeights* output,
                               std::string*) {
            requested.push_back(index);
            *output = weights;
            return true;
        };
    multimodal::FloatMatrix actual;
    std::vector<multimodal::SiglipLayerSummary> summaries;
    multimodal::SiglipEncoderStats stats;
    CHECK(multimodal::run_siglip_encoder(input, 2, 1, 1.0e-5f, provider, post_norm,
                                         &actual, &summaries, &stats, &error));
    CHECK(error.empty());
    CHECK(requested == std::vector<std::size_t>({0, 1}));
    CHECK(summaries.size() == 2);
    CHECK(summaries[0].index == 0 && summaries[1].index == 1);
    CHECK(summaries[0].loaded_weight_bytes > 0);
    CHECK(summaries[0].loaded_weight_bytes == summaries[1].loaded_weight_bytes);
    CHECK(stats.blocks_executed == 2);
    CHECK(stats.peak_block_weight_bytes == summaries[0].loaded_weight_bytes);
    CHECK(actual.rows == expected.rows && actual.cols == expected.cols);
    for (std::size_t i = 0; i < actual.values.size(); ++i) {
        CHECK(near(actual.values[i], expected.values[i]));
    }

    const multimodal::SiglipBlockWeightProvider failing_provider =
        [&weights](std::size_t index, multimodal::SiglipBlockWeights* output, std::string* message) {
            if (index == 1) {
                if (message != nullptr) *message = "synthetic block load failed";
                return false;
            }
            *output = weights;
            return true;
        };
    CHECK(!multimodal::run_siglip_encoder(input, 2, 1, 1.0e-5f, failing_provider,
                                          post_norm, &actual, &summaries, &stats, &error));
    CHECK(error == "synthetic block load failed");
    CHECK(!multimodal::run_siglip_encoder(input, 0, 1, 1.0e-5f, provider,
                                          post_norm, &actual, &summaries, &stats, &error));
}

void test_cblas_matches_scalar_backend() {
    if (!multimodal::vision_compute_backend_available(multimodal::VisionComputeBackend::Cblas)) {
        multimodal::FloatMatrix output;
        std::string error;
        CHECK(!multimodal::apply_linear_with_backend(
            {1, 2, {1.0f, 2.0f}}, identity(2), multimodal::VisionComputeBackend::Cblas,
            &output, &error));
        CHECK(error.find("not enabled") != std::string::npos);
        return;
    }

    const multimodal::PatchTensor patches{
        4, 3, {0.1f, -0.2f, 0.3f,
               0.4f, 0.5f, -0.6f,
               -0.7f, 0.8f, 0.9f,
               1.0f, -1.1f, 1.2f}};
    const auto projection = patterned_linear(3, 4);
    const multimodal::PatchEmbeddingWeights embedding_weights{
        3, 4, 4, projection.weight, projection.bias,
        {0.01f, 0.02f, 0.03f, 0.04f,
         0.05f, 0.06f, 0.07f, 0.08f,
         0.09f, 0.10f, 0.11f, 0.12f,
         0.13f, 0.14f, 0.15f, 0.16f}};
    const std::vector<std::size_t> positions{3, 1, 0, 2};
    multimodal::FloatMatrix scalar_embedding;
    multimodal::FloatMatrix cblas_embedding;
    std::string error;
    CHECK(multimodal::embed_patches_at_positions_with_backend(
        patches, positions, embedding_weights, multimodal::VisionComputeBackend::ScalarReference,
        &scalar_embedding, &error));
    CHECK(multimodal::embed_patches_at_positions_with_backend(
        patches, positions, embedding_weights, multimodal::VisionComputeBackend::Cblas,
        &cblas_embedding, &error));
    CHECK(scalar_embedding.rows == cblas_embedding.rows);
    CHECK(scalar_embedding.cols == cblas_embedding.cols);
    for (std::size_t i = 0; i < scalar_embedding.values.size(); ++i) {
        CHECK(near(scalar_embedding.values[i], cblas_embedding.values[i], 2.0e-6f));
    }

    multimodal::SiglipBlockWeights weights;
    weights.layer_norm_1 = {{1.1f, 0.9f, 1.2f, 0.8f}, {0.01f, -0.02f, 0.03f, -0.04f}};
    weights.query = patterned_linear(4, 4);
    weights.key = patterned_linear(4, 4);
    weights.value = patterned_linear(4, 4);
    weights.attention_output = patterned_linear(4, 4);
    weights.layer_norm_2 = {{0.8f, 1.2f, 0.9f, 1.1f}, {-0.01f, 0.02f, -0.03f, 0.04f}};
    weights.feed_forward_input = patterned_linear(4, 6);
    weights.feed_forward_output = patterned_linear(6, 4);
    multimodal::SiglipBlockTrace scalar_trace;
    multimodal::SiglipBlockTrace cblas_trace;
    CHECK(multimodal::run_siglip_block_with_backend(
        scalar_embedding, 2, 1.0e-5f, weights,
        multimodal::VisionComputeBackend::ScalarReference, &scalar_trace, &error));
    CHECK(multimodal::run_siglip_block_with_backend(
        scalar_embedding, 2, 1.0e-5f, weights,
        multimodal::VisionComputeBackend::Cblas, &cblas_trace, &error));
    CHECK(scalar_trace.output.rows == cblas_trace.output.rows);
    CHECK(scalar_trace.output.cols == cblas_trace.output.cols);
    for (std::size_t i = 0; i < scalar_trace.output.values.size(); ++i) {
        CHECK(near(scalar_trace.output.values[i], cblas_trace.output.values[i], 2.0e-5f));
    }
}

}  // namespace

int main() {
    test_patch_and_position_embedding();
    test_siglip_block_order();
    test_siglip_encoder_streaming_weights();
    test_cblas_matches_scalar_backend();
    if (failures != 0) return 1;
    std::cout << "multimodal vision encoder checks passed\n";
    return 0;
}
