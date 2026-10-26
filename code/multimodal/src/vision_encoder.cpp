// 多模态 CPU：SigLIP 视觉编码器数学实现
// 职责：执行 Patch/Position Embedding、LayerNorm、全注意力、GELU MLP 和 residual。
// 契约：实现只接收 Float32 权重；GGUF/Q8_0 payload 适配由独立加载层处理。
// 验证：multimodal_vision_encoder_test。
#include "multimodal/vision_encoder.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef CPP_LLM_MULTIMODAL_HAS_CBLAS
#include <cblas.h>
#endif

namespace multimodal {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

bool checked_product(std::size_t lhs, std::size_t rhs, std::size_t* result) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) return false;
    *result = lhs * rhs;
    return true;
}

bool prepare_matrix(std::size_t rows,
                    std::size_t cols,
                    FloatMatrix* output,
                    std::string* error) {
    std::size_t elements = 0;
    if (rows == 0 || cols == 0 || !checked_product(rows, cols, &elements)) {
        return fail(error, "matrix shape is invalid or overflows");
    }
    output->rows = rows;
    output->cols = cols;
    output->values.assign(elements, 0.0f);
    return true;
}

bool scalar_linear(const FloatMatrix& input,
                   const LinearWeights& weights,
                   FloatMatrix* output,
                   std::string* error) {
    if (!input.valid() || !weights.valid() || input.cols != weights.input_size) {
        return fail(error, "linear input or weight shape is invalid");
    }
    if (!prepare_matrix(input.rows, weights.output_size, output, error)) return false;
    for (std::size_t row = 0; row < input.rows; ++row) {
        for (std::size_t out = 0; out < weights.output_size; ++out) {
            float value = weights.bias.empty() ? 0.0f : weights.bias[out];
            const std::size_t weight_offset = out * weights.input_size;
            for (std::size_t in = 0; in < weights.input_size; ++in) {
                value += input.at(row, in) * weights.weight[weight_offset + in];
            }
            output->at(row, out) = value;
        }
    }
    return true;
}

#ifdef CPP_LLM_MULTIMODAL_HAS_CBLAS
bool cblas_dimension(std::size_t value) {
    return value <= static_cast<std::size_t>(INT_MAX);
}
#endif

bool linear(const FloatMatrix& input,
            const LinearWeights& weights,
            VisionComputeBackend backend,
            FloatMatrix* output,
            std::string* error) {
    if (backend == VisionComputeBackend::ScalarReference) {
        return scalar_linear(input, weights, output, error);
    }
#ifdef CPP_LLM_MULTIMODAL_HAS_CBLAS
    if (!input.valid() || !weights.valid() || input.cols != weights.input_size) {
        return fail(error, "linear input or weight shape is invalid");
    }
    if (!cblas_dimension(input.rows) || !cblas_dimension(input.cols) ||
        !cblas_dimension(weights.output_size)) {
        return fail(error, "CBLAS linear dimension exceeds int range");
    }
    if (!prepare_matrix(input.rows, weights.output_size, output, error)) return false;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(input.rows), static_cast<int>(weights.output_size),
                static_cast<int>(input.cols), 1.0f, input.values.data(),
                static_cast<int>(input.cols), weights.weight.data(),
                static_cast<int>(weights.input_size), 0.0f, output->values.data(),
                static_cast<int>(weights.output_size));
    if (!weights.bias.empty()) {
        for (std::size_t row = 0; row < output->rows; ++row) {
            for (std::size_t col = 0; col < output->cols; ++col) {
                output->values[row * output->cols + col] += weights.bias[col];
            }
        }
    }
    return true;
#else
    (void)input;
    (void)weights;
    (void)output;
    return fail(error, "CBLAS backend is not enabled in this build");
#endif
}

bool layer_norm_impl(const FloatMatrix& input,
                     const LayerNormWeights& weights,
                     float epsilon,
                     FloatMatrix* output,
                     std::string* error) {
    if (!input.valid() || !weights.valid(input.cols) || !std::isfinite(epsilon) || epsilon <= 0.0f) {
        return fail(error, "layer norm input, weights, or epsilon is invalid");
    }
    *output = input;
    for (std::size_t row = 0; row < input.rows; ++row) {
        float mean = 0.0f;
        for (std::size_t col = 0; col < input.cols; ++col) mean += input.at(row, col);
        mean /= static_cast<float>(input.cols);

        float variance = 0.0f;
        for (std::size_t col = 0; col < input.cols; ++col) {
            const float centered = input.at(row, col) - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(input.cols);
        const float inverse_std = 1.0f / std::sqrt(variance + epsilon);
        for (std::size_t col = 0; col < input.cols; ++col) {
            output->at(row, col) =
                (input.at(row, col) - mean) * inverse_std * weights.weight[col] +
                weights.bias[col];
        }
    }
    return true;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t* result) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) return false;
    *result = lhs + rhs;
    return true;
}

bool add_weight_bytes(const std::vector<float>& values,
                      std::size_t* total,
                      std::string* error) {
    std::size_t bytes = 0;
    if (!checked_product(values.size(), sizeof(float), &bytes) ||
        !checked_add(*total, bytes, total)) {
        return fail(error, "SigLIP block weight size overflows");
    }
    return true;
}

bool block_weight_bytes(const SiglipBlockWeights& weights,
                        std::size_t* bytes,
                        std::string* error) {
    *bytes = 0;
    const auto add_linear = [bytes, error](const LinearWeights& linear_weights) {
        return add_weight_bytes(linear_weights.weight, bytes, error) &&
               add_weight_bytes(linear_weights.bias, bytes, error);
    };
    return add_weight_bytes(weights.layer_norm_1.weight, bytes, error) &&
           add_weight_bytes(weights.layer_norm_1.bias, bytes, error) &&
           add_linear(weights.query) && add_linear(weights.key) &&
           add_linear(weights.value) && add_linear(weights.attention_output) &&
           add_weight_bytes(weights.layer_norm_2.weight, bytes, error) &&
           add_weight_bytes(weights.layer_norm_2.bias, bytes, error) &&
           add_linear(weights.feed_forward_input) &&
           add_linear(weights.feed_forward_output);
}

double matrix_checksum(const FloatMatrix& matrix) {
    double result = 0.0;
    for (std::size_t i = 0; i < matrix.values.size(); ++i) {
        result += static_cast<double>((i % 97) + 1) * matrix.values[i];
    }
    return result;
}

bool add(const FloatMatrix& lhs, const FloatMatrix& rhs, FloatMatrix* output, std::string* error) {
    if (!lhs.valid() || !rhs.valid() || lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
        return fail(error, "residual matrix shapes do not match");
    }
    *output = lhs;
    for (std::size_t i = 0; i < output->values.size(); ++i) output->values[i] += rhs.values[i];
    return true;
}

bool self_attention(const FloatMatrix& input,
                    std::size_t heads,
                    const SiglipBlockWeights& weights,
                    VisionComputeBackend backend,
                    FloatMatrix* output,
                    std::string* error) {
    if (heads == 0 || input.cols % heads != 0) {
        return fail(error, "vision hidden size must be divisible by attention heads");
    }
    FloatMatrix query;
    FloatMatrix key;
    FloatMatrix value;
    if (!linear(input, weights.query, backend, &query, error) ||
        !linear(input, weights.key, backend, &key, error) ||
        !linear(input, weights.value, backend, &value, error)) {
        return false;
    }

    const std::size_t head_dim = input.cols / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    FloatMatrix combined;
    if (!prepare_matrix(input.rows, input.cols, &combined, error)) return false;
    if (backend == VisionComputeBackend::Cblas) {
#ifdef CPP_LLM_MULTIMODAL_HAS_CBLAS
        if (!cblas_dimension(input.rows) || !cblas_dimension(head_dim)) {
            return fail(error, "CBLAS attention dimension exceeds int range");
        }
        const std::size_t head_elements = input.rows * head_dim;
        std::vector<float> packed_query(head_elements);
        std::vector<float> packed_key(head_elements);
        std::vector<float> packed_value(head_elements);
        std::vector<float> scores(input.rows * input.rows);
        std::vector<float> attended(head_elements);
        for (std::size_t head = 0; head < heads; ++head) {
            const std::size_t head_offset = head * head_dim;
            for (std::size_t token = 0; token < input.rows; ++token) {
                const std::size_t source = token * input.cols + head_offset;
                const std::size_t destination = token * head_dim;
                std::copy_n(query.values.data() + source, head_dim,
                            packed_query.data() + destination);
                std::copy_n(key.values.data() + source, head_dim,
                            packed_key.data() + destination);
                std::copy_n(value.values.data() + source, head_dim,
                            packed_value.data() + destination);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        static_cast<int>(input.rows), static_cast<int>(input.rows),
                        static_cast<int>(head_dim), scale, packed_query.data(),
                        static_cast<int>(head_dim), packed_key.data(),
                        static_cast<int>(head_dim), 0.0f, scores.data(),
                        static_cast<int>(input.rows));
            for (std::size_t row = 0; row < input.rows; ++row) {
                float* row_scores = scores.data() + row * input.rows;
                const float maximum = *std::max_element(row_scores, row_scores + input.rows);
                float denominator = 0.0f;
                for (std::size_t col = 0; col < input.rows; ++col) {
                    row_scores[col] = std::exp(row_scores[col] - maximum);
                    denominator += row_scores[col];
                }
                for (std::size_t col = 0; col < input.rows; ++col) {
                    row_scores[col] /= denominator;
                }
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(input.rows), static_cast<int>(head_dim),
                        static_cast<int>(input.rows), 1.0f, scores.data(),
                        static_cast<int>(input.rows), packed_value.data(),
                        static_cast<int>(head_dim), 0.0f, attended.data(),
                        static_cast<int>(head_dim));
            for (std::size_t token = 0; token < input.rows; ++token) {
                std::copy_n(attended.data() + token * head_dim, head_dim,
                            combined.values.data() + token * input.cols + head_offset);
            }
        }
        return linear(combined, weights.attention_output, backend, output, error);
#else
        return fail(error, "CBLAS backend is not enabled in this build");
#endif
    }

    std::vector<float> scores(input.rows, 0.0f);
    for (std::size_t query_token = 0; query_token < input.rows; ++query_token) {
        for (std::size_t head = 0; head < heads; ++head) {
            const std::size_t head_offset = head * head_dim;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t key_token = 0; key_token < input.rows; ++key_token) {
                float score = 0.0f;
                for (std::size_t dim = 0; dim < head_dim; ++dim) {
                    score += query.at(query_token, head_offset + dim) *
                             key.at(key_token, head_offset + dim);
                }
                scores[key_token] = score * scale;
                maximum = std::max(maximum, scores[key_token]);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (std::size_t dim = 0; dim < head_dim; ++dim) {
                float result = 0.0f;
                for (std::size_t key_token = 0; key_token < input.rows; ++key_token) {
                    result += scores[key_token] / denominator * value.at(key_token, head_offset + dim);
                }
                combined.at(query_token, head_offset + dim) = result;
            }
        }
    }
    return linear(combined, weights.attention_output, backend, output, error);
}

float gelu(float value) {
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    return 0.5f * value *
           (1.0f + std::tanh(kSqrtTwoOverPi * (value + 0.044715f * value * value * value)));
}

}  // namespace

bool FloatMatrix::valid() const {
    std::size_t expected = 0;
    return rows > 0 && cols > 0 && checked_product(rows, cols, &expected) && values.size() == expected;
}

float& FloatMatrix::at(std::size_t row, std::size_t col) {
    if (row >= rows || col >= cols) throw std::out_of_range("FloatMatrix index is out of range");
    return values[row * cols + col];
}

float FloatMatrix::at(std::size_t row, std::size_t col) const {
    if (row >= rows || col >= cols) throw std::out_of_range("FloatMatrix index is out of range");
    return values[row * cols + col];
}

bool LinearWeights::valid() const {
    std::size_t expected = 0;
    return input_size > 0 && output_size > 0 &&
           checked_product(input_size, output_size, &expected) && weight.size() == expected &&
           (bias.empty() || bias.size() == output_size);
}

bool LayerNormWeights::valid(std::size_t hidden_size) const {
    return hidden_size > 0 && weight.size() == hidden_size && bias.size() == hidden_size;
}

bool PatchEmbeddingWeights::valid() const {
    std::size_t patch_elements = 0;
    std::size_t position_elements = 0;
    return patch_values > 0 && hidden_size > 0 && position_count > 0 &&
           checked_product(patch_values, hidden_size, &patch_elements) &&
           checked_product(position_count, hidden_size, &position_elements) &&
           patch_weight.size() == patch_elements && patch_bias.size() == hidden_size &&
           position_weight.size() == position_elements;
}

bool SiglipBlockWeights::valid(std::size_t hidden_size) const {
    return layer_norm_1.valid(hidden_size) && layer_norm_2.valid(hidden_size) &&
           query.valid() && key.valid() && value.valid() && attention_output.valid() &&
           feed_forward_input.valid() && feed_forward_output.valid() &&
           query.input_size == hidden_size && query.output_size == hidden_size &&
           key.input_size == hidden_size && key.output_size == hidden_size &&
           value.input_size == hidden_size && value.output_size == hidden_size &&
           attention_output.input_size == hidden_size && attention_output.output_size == hidden_size &&
           feed_forward_input.input_size == hidden_size &&
           feed_forward_output.input_size == feed_forward_input.output_size &&
           feed_forward_output.output_size == hidden_size;
}

bool embed_patches(const PatchTensor& patches,
                   const PatchEmbeddingWeights& weights,
                   FloatMatrix* output,
                   std::string* error) {
    if (!patches.valid()) return fail(error, "patch tensor is invalid");
    std::vector<std::size_t> positions(static_cast<std::size_t>(patches.patch_count));
    for (std::size_t i = 0; i < positions.size(); ++i) positions[i] = i;
    return embed_patches_at_positions(patches, positions, weights, output, error);
}

bool embed_patches_at_positions(const PatchTensor& patches,
                                const std::vector<std::size_t>& positions,
                                const PatchEmbeddingWeights& weights,
                                FloatMatrix* output,
                                std::string* error) {
    return embed_patches_at_positions_with_backend(
        patches, positions, weights, VisionComputeBackend::ScalarReference, output, error);
}

bool embed_patches_at_positions_with_backend(const PatchTensor& patches,
                                             const std::vector<std::size_t>& positions,
                                             const PatchEmbeddingWeights& weights,
                                             VisionComputeBackend backend,
                                             FloatMatrix* output,
                                             std::string* error) {
    if (output == nullptr) return fail(error, "patch embedding output must not be null");
    if (!patches.valid() || !weights.valid() ||
        static_cast<std::size_t>(patches.patch_values) != weights.patch_values) {
        return fail(error, "patch tensor or embedding weights are invalid");
    }
    if (positions.size() != static_cast<std::size_t>(patches.patch_count)) {
        return fail(error, "patch position count does not match patch tensor");
    }
    for (const std::size_t position : positions) {
        if (position >= weights.position_count) {
            return fail(error, "patch position exceeds position embedding count");
        }
    }

    if (backend == VisionComputeBackend::ScalarReference) {
        if (!prepare_matrix(static_cast<std::size_t>(patches.patch_count),
                            weights.hidden_size, output, error)) {
            return false;
        }
        for (std::size_t patch = 0; patch < output->rows; ++patch) {
            for (std::size_t hidden = 0; hidden < output->cols; ++hidden) {
                float value = weights.patch_bias[hidden] +
                              weights.position_weight[positions[patch] * weights.hidden_size + hidden];
                const std::size_t weight_offset = hidden * weights.patch_values;
                for (std::size_t element = 0; element < weights.patch_values; ++element) {
                    value += patches.values[patch * weights.patch_values + element] *
                             weights.patch_weight[weight_offset + element];
                }
                output->at(patch, hidden) = value;
            }
        }
        if (error != nullptr) error->clear();
        return true;
    }

    const FloatMatrix patch_matrix{static_cast<std::size_t>(patches.patch_count),
                                   static_cast<std::size_t>(patches.patch_values), patches.values};
    const LinearWeights patch_projection{weights.patch_values, weights.hidden_size,
                                         weights.patch_weight, weights.patch_bias};
    if (!linear(patch_matrix, patch_projection, backend, output, error)) return false;
    for (std::size_t patch = 0; patch < output->rows; ++patch) {
        for (std::size_t hidden = 0; hidden < output->cols; ++hidden) {
            output->at(patch, hidden) +=
                weights.position_weight[positions[patch] * weights.hidden_size + hidden];
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

bool apply_linear(const FloatMatrix& input,
                  const LinearWeights& weights,
                  FloatMatrix* output,
                  std::string* error) {
    if (output == nullptr) return fail(error, "linear output must not be null");
    return apply_linear_with_backend(
        input, weights, VisionComputeBackend::ScalarReference, output, error);
}

bool apply_linear_with_backend(const FloatMatrix& input,
                               const LinearWeights& weights,
                               VisionComputeBackend backend,
                               FloatMatrix* output,
                               std::string* error) {
    if (output == nullptr) return fail(error, "linear output must not be null");
    const bool success = linear(input, weights, backend, output, error);
    if (success && error != nullptr) error->clear();
    return success;
}

bool run_siglip_block(const FloatMatrix& input,
                      std::size_t attention_heads,
                      float layer_norm_epsilon,
                      const SiglipBlockWeights& weights,
                      SiglipBlockTrace* trace,
                      std::string* error) {
    return run_siglip_block_with_backend(input, attention_heads, layer_norm_epsilon,
                                         weights, VisionComputeBackend::ScalarReference,
                                         trace, error);
}

bool run_siglip_block_with_backend(const FloatMatrix& input,
                                   std::size_t attention_heads,
                                   float layer_norm_epsilon,
                                   const SiglipBlockWeights& weights,
                                   VisionComputeBackend backend,
                                   SiglipBlockTrace* trace,
                                   std::string* error) {
    if (trace == nullptr) return fail(error, "SigLIP block trace must not be null");
    if (!input.valid() || !weights.valid(input.cols)) {
        return fail(error, "SigLIP block input or weights are invalid");
    }

    SiglipBlockTrace result;
    if (!layer_norm_impl(input, weights.layer_norm_1, layer_norm_epsilon,
                         &result.normalized_attention_input, error) ||
        !self_attention(result.normalized_attention_input, attention_heads, weights, backend,
                        &result.attention_output, error) ||
        !add(input, result.attention_output, &result.attention_residual, error) ||
        !layer_norm_impl(result.attention_residual, weights.layer_norm_2, layer_norm_epsilon,
                         &result.normalized_feed_forward_input, error) ||
        !linear(result.normalized_feed_forward_input, weights.feed_forward_input, backend,
                &result.feed_forward_hidden, error)) {
        return false;
    }
    for (float& value : result.feed_forward_hidden.values) value = gelu(value);
    FloatMatrix feed_forward_output;
    if (!linear(result.feed_forward_hidden, weights.feed_forward_output, backend,
                &feed_forward_output, error) ||
        !add(result.attention_residual, feed_forward_output, &result.output, error)) {
        return false;
    }

    *trace = std::move(result);
    if (error != nullptr) error->clear();
    return true;
}

bool apply_layer_norm(const FloatMatrix& input,
                      const LayerNormWeights& weights,
                      float epsilon,
                      FloatMatrix* output,
                      std::string* error) {
    if (output == nullptr) return fail(error, "layer norm output must not be null");
    const bool success = layer_norm_impl(input, weights, epsilon, output, error);
    if (success && error != nullptr) error->clear();
    return success;
}

bool run_siglip_encoder(const FloatMatrix& input,
                        std::size_t block_count,
                        std::size_t attention_heads,
                        float layer_norm_epsilon,
                        const SiglipBlockWeightProvider& weight_provider,
                        const LayerNormWeights& post_layer_norm,
                        FloatMatrix* output,
                        std::vector<SiglipLayerSummary>* summaries,
                        SiglipEncoderStats* stats,
                        std::string* error) {
    return run_siglip_encoder_with_backend(
        input, block_count, attention_heads, layer_norm_epsilon, weight_provider,
        post_layer_norm, VisionComputeBackend::ScalarReference,
        output, summaries, stats, error);
}

bool run_siglip_encoder_with_backend(const FloatMatrix& input,
                                     std::size_t block_count,
                                     std::size_t attention_heads,
                                     float layer_norm_epsilon,
                                     const SiglipBlockWeightProvider& weight_provider,
                                     const LayerNormWeights& post_layer_norm,
                                     VisionComputeBackend backend,
                                     FloatMatrix* output,
                                     std::vector<SiglipLayerSummary>* summaries,
                                     SiglipEncoderStats* stats,
                                     std::string* error) {
    if (output == nullptr || summaries == nullptr || stats == nullptr) {
        return fail(error, "SigLIP encoder outputs must not be null");
    }
    if (!input.valid() || block_count == 0 || !weight_provider ||
        !post_layer_norm.valid(input.cols)) {
        return fail(error, "SigLIP encoder input, block count, provider, or post norm is invalid");
    }

    FloatMatrix current = input;
    std::vector<SiglipLayerSummary> layer_summaries;
    layer_summaries.reserve(block_count);
    SiglipEncoderStats encoder_stats;
    for (std::size_t block = 0; block < block_count; ++block) {
        SiglipBlockWeights weights;
        if (!weight_provider(block, &weights, error)) return false;
        if (!weights.valid(input.cols)) {
            return fail(error, "SigLIP provider returned invalid weights for block " +
                                   std::to_string(block));
        }
        std::size_t loaded_bytes = 0;
        if (!block_weight_bytes(weights, &loaded_bytes, error)) return false;
        encoder_stats.peak_block_weight_bytes =
            std::max(encoder_stats.peak_block_weight_bytes, loaded_bytes);

        SiglipBlockTrace trace;
        if (!run_siglip_block_with_backend(current, attention_heads, layer_norm_epsilon,
                                           weights, backend, &trace, error)) {
            return false;
        }
        layer_summaries.push_back(
            {block, matrix_checksum(trace.output), trace.output.values.front(), loaded_bytes});
        current = std::move(trace.output);
        ++encoder_stats.blocks_executed;
    }

    FloatMatrix normalized;
    if (!layer_norm_impl(current, post_layer_norm, layer_norm_epsilon, &normalized, error)) {
        return false;
    }
    *output = std::move(normalized);
    *summaries = std::move(layer_summaries);
    *stats = encoder_stats;
    if (error != nullptr) error->clear();
    return true;
}

bool vision_compute_backend_available(VisionComputeBackend backend) {
    if (backend == VisionComputeBackend::ScalarReference) return true;
#ifdef CPP_LLM_MULTIMODAL_HAS_CBLAS
    return true;
#else
    return false;
#endif
}

}  // namespace multimodal
