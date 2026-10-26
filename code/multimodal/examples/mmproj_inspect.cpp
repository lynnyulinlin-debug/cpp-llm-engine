// 多模态 CPU：mmproj schema 检查工具
// 职责：读取真实 mmproj GGUF，打印视觉配置并执行完整 tensor schema 校验。
// 契约：只读取 metadata 和 tensor index，不加载 109 MiB 权重 payload。
// 运行：multimodal_mmproj_inspect <mmproj.gguf>。
#include "multimodal/mmproj_schema.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: multimodal_mmproj_inspect <mmproj.gguf>\n";
        return 2;
    }

    chapter01::GgufModelIndex index;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error)) {
        std::cerr << "read failed: " << error << '\n';
        return 1;
    }

    multimodal::VisionModelConfig config;
    if (!multimodal::MmprojSchema::config_from_gguf(index.metadata, &config, &error)) {
        std::cerr << "config failed: " << error << '\n';
        return 1;
    }

    multimodal::VisionTensorLayout layout;
    if (!multimodal::MmprojSchema::validate_gguf(config, index, &layout, &error)) {
        std::cerr << "schema failed: " << error << '\n';
        return 1;
    }

    std::cout << "architecture=" << config.architecture << '\n'
              << "projector=" << config.projector_type << '\n'
              << "image=" << config.image_size << " patch=" << config.patch_size
              << " hidden=" << config.hidden_size << " ffn=" << config.intermediate_size
              << " blocks=" << config.block_count << " heads=" << config.attention_head_count
              << " output=" << config.projection_dim << " merge=" << config.merge_size << '\n'
              << "mean=" << config.image_mean[0] << ',' << config.image_mean[1] << ','
              << config.image_mean[2] << " std=" << config.image_std[0] << ','
              << config.image_std[1] << ',' << config.image_std[2] << '\n'
              << "tensors=" << index.tensors.size()
              << " expected=" << layout.expected_tensor_count << '\n'
              << "first_block=" << layout.blocks.front().attention_query_weight << '\n'
              << "schema=valid\n";
    return 0;
}
