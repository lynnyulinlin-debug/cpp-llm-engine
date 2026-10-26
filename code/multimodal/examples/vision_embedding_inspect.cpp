// 多模态 CPU：真实 mmproj Patch Embedding 检查工具
// 职责：加载 F32 patch/position 权重，对真实图片的第一个 patch 计算 embedding。
// 契约：只验证 MM-P4A 的 embedding payload，不加载 Q8_0 SigLIP block。
// 运行：multimodal_vision_embedding_inspect <mmproj.gguf> <image>。
#include "multimodal/image.h"
#include "multimodal/image_preprocessor.h"
#include "multimodal/mmproj_schema.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_weights.h"

#include <cstddef>
#include <iostream>
#include <limits>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: multimodal_vision_embedding_inspect <mmproj.gguf> <image>\n";
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

    multimodal::PatchEmbeddingWeights weights;
    if (!multimodal::load_patch_embedding_weights(
            argv[1], index, config, layout, &weights, &error)) {
        std::cerr << "weight load failed: " << error << '\n';
        return 1;
    }

    multimodal::RgbImage image;
    if (!multimodal::load_rgb_image(argv[2], &image, &error)) {
        std::cerr << "image load failed: " << error << '\n';
        return 1;
    }
    const multimodal::ImagePreprocessorConfig preprocess{
        static_cast<int>(config.image_size),
        static_cast<int>(config.image_size),
        static_cast<int>(config.patch_size),
        static_cast<int>(config.patch_size),
        config.image_mean,
        config.image_std};
    multimodal::PreprocessedImage prepared;
    if (!multimodal::preprocess_image(image, preprocess, &prepared, &error)) {
        std::cerr << "preprocess failed: " << error << '\n';
        return 1;
    }
    if (prepared.patches.patch_values <= 0 ||
        prepared.patches.values.size() <
            static_cast<std::size_t>(prepared.patches.patch_values)) {
        std::cerr << "preprocess failed: first patch payload is incomplete\n";
        return 1;
    }

    multimodal::PatchTensor first_patch;
    first_patch.patch_count = 1;
    first_patch.patch_values = prepared.patches.patch_values;
    first_patch.values.assign(prepared.patches.values.begin(),
                              prepared.patches.values.begin() + prepared.patches.patch_values);
    multimodal::FloatMatrix embedding;
    if (!multimodal::embed_patches(first_patch, weights, &embedding, &error)) {
        std::cerr << "embedding failed: " << error << '\n';
        return 1;
    }

    double checksum = 0.0;
    for (std::size_t i = 0; i < embedding.values.size(); ++i) {
        checksum += static_cast<double>(i + 1) * embedding.values[i];
    }
    std::cout << "embedding=" << embedding.rows << 'x' << embedding.cols << '\n'
              << "patch_values=" << first_patch.patch_values << '\n'
              << "first=" << embedding.values.front() << '\n'
              << "weighted_checksum=" << checksum << '\n'
              << "payload=f32-valid\n";
    return 0;
}
