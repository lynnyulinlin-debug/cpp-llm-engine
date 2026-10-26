// 多模态 CPU：视觉模型 GGUF schema 接口
// 职责：把 mmproj metadata 和 tensor index 转换为可执行前的视觉模型契约。
// 契约：只验证配置、名称和 shape，不读取或解码 tensor payload。
// 验证：multimodal_mmproj_schema_test 与真实 SmolVLM mmproj inspect。
#pragma once

#include "gguf_reader.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace multimodal {

struct VisionModelConfig {
    std::string architecture;
    std::string projector_type;
    std::size_t image_size = 0;
    std::size_t patch_size = 0;
    std::size_t hidden_size = 0;
    std::size_t intermediate_size = 0;
    std::size_t block_count = 0;
    std::size_t attention_head_count = 0;
    std::size_t projection_dim = 0;
    std::size_t merge_size = 0;
    float layer_norm_epsilon = 0.0f;
    std::array<float, 3> image_mean{};
    std::array<float, 3> image_std{};
};

struct VisionBlockLayout {
    std::size_t index = 0;
    std::string layer_norm_1_weight;
    std::string layer_norm_1_bias;
    std::string layer_norm_2_weight;
    std::string layer_norm_2_bias;
    std::string attention_query_weight;
    std::string attention_query_bias;
    std::string attention_key_weight;
    std::string attention_key_bias;
    std::string attention_value_weight;
    std::string attention_value_bias;
    std::string attention_output_weight;
    std::string attention_output_bias;
    std::string feed_forward_up_weight;
    std::string feed_forward_up_bias;
    std::string feed_forward_down_weight;
    std::string feed_forward_down_bias;
};

struct VisionTensorLayout {
    std::string patch_embedding_weight;
    std::string patch_embedding_bias;
    std::string position_embedding_weight;
    std::vector<VisionBlockLayout> blocks;
    std::string post_layer_norm_weight;
    std::string post_layer_norm_bias;
    std::string projector_weight;
    std::size_t expected_tensor_count = 0;
};

class MmprojSchema {
public:
    static bool config_from_gguf(const chapter01::MetadataMap& metadata,
                                 VisionModelConfig* config,
                                 std::string* error = nullptr);

    static bool validate_gguf(const VisionModelConfig& config,
                              const chapter01::GgufModelIndex& index,
                              VisionTensorLayout* layout,
                              std::string* error = nullptr);
};

}  // namespace multimodal
