// 多模态 CPU：真实 mmproj 第 0 个 SigLIP Block 检查工具
// 职责：对真实图片前 N 个 patch 执行 F32 embedding、Q8_0 权重解码和单 Block。
// 契约：N 默认 4，限制为 1..32；输出中间 checksum，不替代 1024-token 完整视觉编码。
// 运行：multimodal_vision_block_inspect <mmproj.gguf> <image> [patch-count]。
#include "multimodal/image.h"
#include "multimodal/image_preprocessor.h"
#include "multimodal/mmproj_schema.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_weights.h"

#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

double checksum(const multimodal::FloatMatrix& matrix) {
    double result = 0.0;
    for (std::size_t i = 0; i < matrix.values.size(); ++i) {
        result += static_cast<double>((i % 97) + 1) * matrix.values[i];
    }
    return result;
}

double checksum(const std::vector<float>& values) {
    double result = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        result += static_cast<double>((i % 97) + 1) * values[i];
    }
    return result;
}

bool parse_patch_count(const char* value, std::size_t* result) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 || parsed > 32) return false;
    *result = static_cast<std::size_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: multimodal_vision_block_inspect <mmproj.gguf> <image> [patch-count]\n";
        return 2;
    }
    std::size_t patch_count = 4;
    if (argc == 4 && !parse_patch_count(argv[3], &patch_count)) {
        std::cerr << "patch-count must be an integer in [1, 32]\n";
        return 2;
    }

    chapter01::GgufModelIndex index;
    multimodal::VisionModelConfig config;
    multimodal::VisionTensorLayout layout;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error) ||
        !multimodal::MmprojSchema::config_from_gguf(index.metadata, &config, &error) ||
        !multimodal::MmprojSchema::validate_gguf(config, index, &layout, &error)) {
        std::cerr << "mmproj failed: " << error << '\n';
        return 1;
    }
    if (config.image_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        config.patch_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        std::cerr << "mmproj failed: image or patch size exceeds preprocessor range\n";
        return 1;
    }

    multimodal::PatchEmbeddingWeights embedding_weights;
    multimodal::SiglipBlockWeights block_weights;
    if (!multimodal::load_patch_embedding_weights(
            argv[1], index, config, layout, &embedding_weights, &error) ||
        !multimodal::load_siglip_block_weights(
            argv[1], index, config, layout, 0, &block_weights, &error)) {
        std::cerr << "weight load failed: " << error << '\n';
        return 1;
    }

    multimodal::RgbImage image;
    if (!multimodal::load_rgb_image(argv[2], &image, &error)) {
        std::cerr << "image load failed: " << error << '\n';
        return 1;
    }
    const multimodal::ImagePreprocessorConfig preprocess{
        static_cast<int>(config.image_size), static_cast<int>(config.image_size),
        static_cast<int>(config.patch_size), static_cast<int>(config.patch_size),
        config.image_mean, config.image_std};
    multimodal::PreprocessedImage prepared;
    if (!multimodal::preprocess_image(image, preprocess, &prepared, &error) ||
        patch_count > static_cast<std::size_t>(prepared.patches.patch_count)) {
        std::cerr << "preprocess failed: "
                  << (error.empty() ? "requested patch count exceeds image patches" : error) << '\n';
        return 1;
    }

    multimodal::PatchTensor selected;
    selected.patch_count = static_cast<int>(patch_count);
    selected.patch_values = prepared.patches.patch_values;
    const std::size_t selected_values =
        patch_count * static_cast<std::size_t>(selected.patch_values);
    selected.values.assign(prepared.patches.values.begin(),
                           prepared.patches.values.begin() + selected_values);

    multimodal::FloatMatrix embedding;
    multimodal::SiglipBlockTrace trace;
    if (!multimodal::embed_patches(selected, embedding_weights, &embedding, &error) ||
        !multimodal::run_siglip_block(embedding, config.attention_head_count,
                                      config.layer_norm_epsilon,
                                      block_weights, &trace, &error)) {
        std::cerr << "vision block failed: " << error << '\n';
        return 1;
    }

    std::cout << std::setprecision(10)
              << "patches=" << patch_count << " hidden=" << config.hidden_size << '\n'
              << "patch_checksum=" << checksum(selected.values) << '\n'
              << "embedding_checksum=" << checksum(embedding) << '\n'
              << "embedding_first=" << embedding.values.front() << '\n'
              << "norm1_checksum=" << checksum(trace.normalized_attention_input) << '\n'
              << "attention_checksum=" << checksum(trace.attention_output) << '\n'
              << "residual1_checksum=" << checksum(trace.attention_residual) << '\n'
              << "norm2_checksum=" << checksum(trace.normalized_feed_forward_input) << '\n'
              << "ffn_checksum=" << checksum(trace.feed_forward_hidden) << '\n'
              << "output_checksum=" << checksum(trace.output) << '\n'
              << "first=" << trace.output.values.front() << '\n'
              << "block=q8_0-valid\n";
    return 0;
}
