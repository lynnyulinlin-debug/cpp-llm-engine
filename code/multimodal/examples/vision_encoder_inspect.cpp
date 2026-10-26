// 多模态 CPU：真实 mmproj 12 层 SigLIP Vision Encoder 检查工具
// 职责：对图片前 N 个 patch 逐层加载 Q8_0 权重，执行 12 个 Block 和 post LayerNorm。
// 契约：N 默认 4、限制 1..32；每层权重在下一层加载前离开作用域。
// 运行：multimodal_vision_encoder_inspect <mmproj.gguf> <image> [patch-count]。
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

namespace {

double checksum(const multimodal::FloatMatrix& matrix) {
    double result = 0.0;
    for (std::size_t i = 0; i < matrix.values.size(); ++i) {
        result += static_cast<double>((i % 97) + 1) * matrix.values[i];
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
        std::cerr << "usage: multimodal_vision_encoder_inspect <mmproj.gguf> <image> [patch-count]\n";
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
    multimodal::LayerNormWeights post_norm;
    if (!multimodal::load_patch_embedding_weights(
            argv[1], index, config, layout, &embedding_weights, &error) ||
        !multimodal::load_post_layer_norm_weights(
            argv[1], index, config, layout, &post_norm, &error)) {
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
    if (!multimodal::embed_patches(selected, embedding_weights, &embedding, &error)) {
        std::cerr << "embedding failed: " << error << '\n';
        return 1;
    }

    const std::string mmproj_path = argv[1];
    const multimodal::SiglipBlockWeightProvider provider =
        [&mmproj_path, &index, &config, &layout](
            std::size_t block, multimodal::SiglipBlockWeights* weights, std::string* message) {
            return multimodal::load_siglip_block_weights(
                mmproj_path, index, config, layout, block, weights, message);
        };
    multimodal::FloatMatrix output;
    std::vector<multimodal::SiglipLayerSummary> summaries;
    multimodal::SiglipEncoderStats stats;
    if (!multimodal::run_siglip_encoder(
            embedding, config.block_count, config.attention_head_count,
            config.layer_norm_epsilon, provider, post_norm,
            &output, &summaries, &stats, &error)) {
        std::cerr << "vision encoder failed: " << error << '\n';
        return 1;
    }

    std::cout << std::setprecision(10)
              << "patches=" << patch_count << " hidden=" << config.hidden_size << '\n';
    for (const auto& summary : summaries) {
        std::cout << "layer=" << summary.index
                  << " checksum=" << summary.output_checksum
                  << " first=" << summary.first_value
                  << " weight_bytes=" << summary.loaded_weight_bytes << '\n';
    }
    std::cout << "blocks_executed=" << stats.blocks_executed << '\n'
              << "peak_block_weight_bytes=" << stats.peak_block_weight_bytes << '\n'
              << "post_norm_checksum=" << checksum(output) << '\n'
              << "first=" << output.values.front() << '\n'
              << "encoder=q8_0-valid\n";
    return 0;
}
