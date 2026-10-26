// 多模态 CPU：SigLIP 视觉编码器数学接口
// 职责：声明 Patch/Position Embedding、SigLIP block 和逐层加载的 Vision Encoder。
// 契约：矩阵使用 row-major，Linear 权重使用 GGUF 的 output-major [output, input] 解释。
// 验证：multimodal_vision_encoder_test。
#pragma once

#include "multimodal/image_preprocessor.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace multimodal {

struct FloatMatrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> values;

    bool valid() const;
    float& at(std::size_t row, std::size_t col);
    float at(std::size_t row, std::size_t col) const;
};

struct LinearWeights {
    std::size_t input_size = 0;
    std::size_t output_size = 0;
    std::vector<float> weight;
    std::vector<float> bias;

    bool valid() const;
};

struct LayerNormWeights {
    std::vector<float> weight;
    std::vector<float> bias;

    bool valid(std::size_t hidden_size) const;
};

struct PatchEmbeddingWeights {
    std::size_t patch_values = 0;
    std::size_t hidden_size = 0;
    std::size_t position_count = 0;
    std::vector<float> patch_weight;
    std::vector<float> patch_bias;
    std::vector<float> position_weight;

    bool valid() const;
};

struct SiglipBlockWeights {
    LayerNormWeights layer_norm_1;
    LinearWeights query;
    LinearWeights key;
    LinearWeights value;
    LinearWeights attention_output;
    LayerNormWeights layer_norm_2;
    LinearWeights feed_forward_input;
    LinearWeights feed_forward_output;

    bool valid(std::size_t hidden_size) const;
};

struct SiglipBlockTrace {
    FloatMatrix normalized_attention_input;
    FloatMatrix attention_output;
    FloatMatrix attention_residual;
    FloatMatrix normalized_feed_forward_input;
    FloatMatrix feed_forward_hidden;
    FloatMatrix output;
};

struct SiglipLayerSummary {
    std::size_t index = 0;
    double output_checksum = 0.0;
    float first_value = 0.0f;
    std::size_t loaded_weight_bytes = 0;
};

struct SiglipEncoderStats {
    std::size_t blocks_executed = 0;
    std::size_t peak_block_weight_bytes = 0;
};

enum class VisionComputeBackend {
    ScalarReference,
    Cblas,
};

using SiglipBlockWeightProvider =
    std::function<bool(std::size_t, SiglipBlockWeights*, std::string*)>;

bool embed_patches(const PatchTensor& patches,
                   const PatchEmbeddingWeights& weights,
                   FloatMatrix* output,
                   std::string* error = nullptr);

bool embed_patches_at_positions(const PatchTensor& patches,
                                const std::vector<std::size_t>& positions,
                                const PatchEmbeddingWeights& weights,
                                FloatMatrix* output,
                                std::string* error = nullptr);

bool embed_patches_at_positions_with_backend(const PatchTensor& patches,
                                             const std::vector<std::size_t>& positions,
                                             const PatchEmbeddingWeights& weights,
                                             VisionComputeBackend backend,
                                             FloatMatrix* output,
                                             std::string* error = nullptr);

bool apply_linear(const FloatMatrix& input,
                  const LinearWeights& weights,
                  FloatMatrix* output,
                  std::string* error = nullptr);

bool apply_linear_with_backend(const FloatMatrix& input,
                               const LinearWeights& weights,
                               VisionComputeBackend backend,
                               FloatMatrix* output,
                               std::string* error = nullptr);

bool run_siglip_block(const FloatMatrix& input,
                      std::size_t attention_heads,
                      float layer_norm_epsilon,
                      const SiglipBlockWeights& weights,
                      SiglipBlockTrace* trace,
                      std::string* error = nullptr);

bool run_siglip_block_with_backend(const FloatMatrix& input,
                                   std::size_t attention_heads,
                                   float layer_norm_epsilon,
                                   const SiglipBlockWeights& weights,
                                   VisionComputeBackend backend,
                                   SiglipBlockTrace* trace,
                                   std::string* error = nullptr);

bool apply_layer_norm(const FloatMatrix& input,
                      const LayerNormWeights& weights,
                      float epsilon,
                      FloatMatrix* output,
                      std::string* error = nullptr);

bool run_siglip_encoder(const FloatMatrix& input,
                        std::size_t block_count,
                        std::size_t attention_heads,
                        float layer_norm_epsilon,
                        const SiglipBlockWeightProvider& weight_provider,
                        const LayerNormWeights& post_layer_norm,
                        FloatMatrix* output,
                        std::vector<SiglipLayerSummary>* summaries,
                        SiglipEncoderStats* stats,
                        std::string* error = nullptr);

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
                                     std::string* error = nullptr);

bool vision_compute_backend_available(VisionComputeBackend backend);

}  // namespace multimodal
