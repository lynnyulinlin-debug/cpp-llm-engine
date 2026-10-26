// 多模态 CPU：mmproj schema 测试
// 职责：用合成 GGUF index 验证视觉配置、198 个 tensor 和失败路径。
// 契约：测试不依赖真实模型、网络、llama.cpp 或 GPU。
// 运行：由 CTest 执行 multimodal_mmproj_schema_test。
#include "multimodal/mmproj_schema.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
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

chapter01::MetadataMap make_metadata() {
    return {
        {"general.architecture", std::string("clip")},
        {"clip.projector_type", std::string("idefics3")},
        {"clip.vision.image_size", std::uint64_t{512}},
        {"clip.vision.patch_size", std::uint64_t{16}},
        {"clip.vision.embedding_length", std::uint64_t{768}},
        {"clip.vision.feed_forward_length", std::uint64_t{3072}},
        {"clip.vision.block_count", std::uint64_t{12}},
        {"clip.vision.attention.head_count", std::uint64_t{12}},
        {"clip.vision.projection_dim", std::uint64_t{960}},
        {"clip.vision.projector.scale_factor", std::uint64_t{4}},
        {"clip.vision.attention.layer_norm_epsilon", 1.0e-6f},
        {"clip.vision.image_mean", std::vector<float>{0.5f, 0.5f, 0.5f}},
        {"clip.vision.image_std", std::vector<float>{0.5f, 0.5f, 0.5f}},
    };
}

void add_tensor(chapter01::GgufModelIndex* index,
                std::string name,
                std::vector<std::uint64_t> dimensions) {
    index->tensors.push_back({std::move(name), std::move(dimensions), 0, 0});
}

chapter01::GgufModelIndex make_index() {
    chapter01::GgufModelIndex index;
    index.version = 3;
    index.metadata = make_metadata();
    add_tensor(&index, "v.patch_embd.weight", {16, 16, 3, 768});
    add_tensor(&index, "v.patch_embd.bias", {768});
    add_tensor(&index, "v.position_embd.weight", {768, 1024});
    for (std::size_t block = 0; block < 12; ++block) {
        const std::string prefix = "v.blk." + std::to_string(block) + ".";
        add_tensor(&index, prefix + "ln1.weight", {768});
        add_tensor(&index, prefix + "ln1.bias", {768});
        add_tensor(&index, prefix + "ln2.weight", {768});
        add_tensor(&index, prefix + "ln2.bias", {768});
        add_tensor(&index, prefix + "attn_q.weight", {768, 768});
        add_tensor(&index, prefix + "attn_q.bias", {768});
        add_tensor(&index, prefix + "attn_k.weight", {768, 768});
        add_tensor(&index, prefix + "attn_k.bias", {768});
        add_tensor(&index, prefix + "attn_v.weight", {768, 768});
        add_tensor(&index, prefix + "attn_v.bias", {768});
        add_tensor(&index, prefix + "attn_out.weight", {768, 768});
        add_tensor(&index, prefix + "attn_out.bias", {768});
        add_tensor(&index, prefix + "ffn_up.weight", {3072, 768});
        add_tensor(&index, prefix + "ffn_up.bias", {768});
        add_tensor(&index, prefix + "ffn_down.weight", {768, 3072});
        add_tensor(&index, prefix + "ffn_down.bias", {3072});
    }
    add_tensor(&index, "v.post_ln.weight", {768});
    add_tensor(&index, "v.post_ln.bias", {768});
    add_tensor(&index, "mm.model.fc.weight", {12288, 960});
    return index;
}

void test_valid_smolvlm_schema() {
    auto index = make_index();
    CHECK(index.tensors.size() == 198);

    multimodal::VisionModelConfig config;
    multimodal::VisionTensorLayout layout;
    std::string error;
    CHECK(multimodal::MmprojSchema::config_from_gguf(index.metadata, &config, &error));
    CHECK(error.empty());
    CHECK(config.image_size == 512);
    CHECK(config.patch_size == 16);
    CHECK(config.hidden_size == 768);
    CHECK(config.block_count == 12);
    CHECK(config.projection_dim == 960);
    CHECK(config.merge_size == 4);
    CHECK(multimodal::MmprojSchema::validate_gguf(config, index, &layout, &error));
    CHECK(error.empty());
    CHECK(layout.expected_tensor_count == 198);
    CHECK(layout.blocks.size() == 12);
    CHECK(layout.blocks.front().attention_query_weight == "v.blk.0.attn_q.weight");
    CHECK(layout.blocks.back().feed_forward_down_weight == "v.blk.11.ffn_down.weight");
}

void test_metadata_errors() {
    auto metadata = make_metadata();
    multimodal::VisionModelConfig config;
    std::string error;

    metadata["clip.projector_type"] = std::string("mlp");
    CHECK(!multimodal::MmprojSchema::config_from_gguf(metadata, &config, &error));
    CHECK(error.find("unsupported") != std::string::npos);

    metadata = make_metadata();
    metadata["clip.vision.image_std"] = std::vector<float>{0.5f, 0.5f};
    CHECK(!multimodal::MmprojSchema::config_from_gguf(metadata, &config, &error));
    CHECK(error.find("three floats") != std::string::npos);

    metadata = make_metadata();
    metadata["clip.vision.attention.head_count"] = std::uint64_t{7};
    CHECK(!multimodal::MmprojSchema::config_from_gguf(metadata, &config, &error));
    CHECK(error.find("divisible") != std::string::npos);

    CHECK(!multimodal::MmprojSchema::config_from_gguf(metadata, nullptr, &error));
}

void test_tensor_errors() {
    auto index = make_index();
    multimodal::VisionModelConfig config;
    multimodal::VisionTensorLayout layout;
    std::string error;
    CHECK(multimodal::MmprojSchema::config_from_gguf(index.metadata, &config, &error));

    auto missing = index;
    missing.tensors.pop_back();
    CHECK(!multimodal::MmprojSchema::validate_gguf(config, missing, &layout, &error));
    CHECK(error.find("count mismatch") != std::string::npos);

    auto duplicate = index;
    duplicate.tensors.back() = duplicate.tensors.front();
    CHECK(!multimodal::MmprojSchema::validate_gguf(config, duplicate, &layout, &error));
    CHECK(error.find("duplicate") != std::string::npos);

    auto wrong_shape = index;
    wrong_shape.tensors.front().dimensions = {16, 16, 3, 767};
    CHECK(!multimodal::MmprojSchema::validate_gguf(config, wrong_shape, &layout, &error));
    CHECK(error.find("shape mismatch") != std::string::npos);

    auto invalid_config = config;
    invalid_config.patch_size = 0;
    CHECK(!multimodal::MmprojSchema::validate_gguf(invalid_config, index, &layout, &error));
    CHECK(error.find("non-zero") != std::string::npos);

    invalid_config = config;
    invalid_config.block_count = std::numeric_limits<std::size_t>::max();
    CHECK(!multimodal::MmprojSchema::validate_gguf(invalid_config, index, &layout, &error));
    CHECK(error.find("overflow") != std::string::npos);

    CHECK(!multimodal::MmprojSchema::validate_gguf(config, index, nullptr, &error));
}

}  // namespace

int main() {
    test_valid_smolvlm_schema();
    test_metadata_errors();
    test_tensor_errors();
    if (failures != 0) return 1;
    std::cout << "multimodal mmproj schema checks passed\n";
    return 0;
}
