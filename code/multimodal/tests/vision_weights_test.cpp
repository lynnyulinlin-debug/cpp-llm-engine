// 多模态 CPU：视觉 GGUF 权重适配测试
// 职责：用小型 F32/Q8_0 payload 验证 Linear shape、解码和 IDEFICS3 FFN 名称交换。
// 契约：测试不依赖真实 mmproj、Python、网络、llama.cpp 或 GPU。
// 运行：由 CTest 执行 multimodal_vision_weights_test。
#include "multimodal/vision_weights.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>

namespace {

#define CHECK(condition)                                                                    \
    do {                                                                                    \
        if (!(condition)) throw std::runtime_error("CHECK failed at line " +              \
                                                   std::to_string(__LINE__) + ": " #condition); \
    } while (false)

void append_f32(std::ofstream& output,
                chapter01::GgufModelIndex* index,
                const std::string& name,
                const std::vector<std::uint64_t>& dimensions,
                std::size_t elements,
                float value) {
    const std::uint64_t offset = static_cast<std::uint64_t>(output.tellp());
    const std::vector<float> values(elements, value);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    index->tensors.push_back({name, dimensions, 0, offset});
}

void append_q8_0(std::ofstream& output,
                 chapter01::GgufModelIndex* index,
                 const std::string& name,
                 const std::vector<std::uint64_t>& dimensions,
                 std::size_t elements,
                 std::int8_t quant) {
    CHECK(elements % 32 == 0);
    const std::uint64_t offset = static_cast<std::uint64_t>(output.tellp());
    for (std::size_t block = 0; block < elements / 32; ++block) {
        const std::uint16_t scale = 0x3c00;  // FP16 1.0
        output.write(reinterpret_cast<const char*>(&scale), sizeof(scale));
        for (std::size_t value = 0; value < 32; ++value) {
            output.write(reinterpret_cast<const char*>(&quant), sizeof(quant));
        }
    }
    index->tensors.push_back({name, dimensions, 8, offset});
}

bool near(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1.0e-6f;
}

}  // namespace

int main() {
    constexpr std::size_t hidden = 32;
    constexpr std::size_t intermediate = 64;
    const auto path = std::filesystem::temp_directory_path() / "multimodal_vision_weights.bin";
    std::filesystem::remove(path);

    chapter01::GgufModelIndex index;
    index.data_offset = 0;
    multimodal::VisionBlockLayout block{
        0,
        "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
        "q.weight", "q.bias", "k.weight", "k.bias",
        "v.weight", "v.bias", "out.weight", "out.bias",
        "legacy_up.weight", "legacy_up.bias",
        "legacy_down.weight", "legacy_down.bias"};

    {
        std::ofstream output(path, std::ios::binary);
        CHECK(output);
        append_f32(output, &index, block.layer_norm_1_weight, {hidden}, hidden, 1.0f);
        append_f32(output, &index, block.layer_norm_1_bias, {hidden}, hidden, 0.0f);
        append_f32(output, &index, block.layer_norm_2_weight, {hidden}, hidden, 2.0f);
        append_f32(output, &index, block.layer_norm_2_bias, {hidden}, hidden, 0.5f);
        append_q8_0(output, &index, block.attention_query_weight, {hidden, hidden}, hidden * hidden, 1);
        append_f32(output, &index, block.attention_query_bias, {hidden}, hidden, 0.1f);
        append_q8_0(output, &index, block.attention_key_weight, {hidden, hidden}, hidden * hidden, 2);
        append_f32(output, &index, block.attention_key_bias, {hidden}, hidden, 0.2f);
        append_q8_0(output, &index, block.attention_value_weight, {hidden, hidden}, hidden * hidden, 3);
        append_f32(output, &index, block.attention_value_bias, {hidden}, hidden, 0.3f);
        append_q8_0(output, &index, block.attention_output_weight, {hidden, hidden}, hidden * hidden, 4);
        append_f32(output, &index, block.attention_output_bias, {hidden}, hidden, 0.4f);
        append_q8_0(output, &index, block.feed_forward_down_weight,
                    {hidden, intermediate}, hidden * intermediate, 5);
        append_f32(output, &index, block.feed_forward_down_bias,
                   {intermediate}, intermediate, 0.5f);
        append_q8_0(output, &index, block.feed_forward_up_weight,
                    {intermediate, hidden}, intermediate * hidden, 6);
        append_f32(output, &index, block.feed_forward_up_bias, {hidden}, hidden, 0.6f);
    }

    multimodal::VisionModelConfig config;
    config.hidden_size = hidden;
    config.intermediate_size = intermediate;
    multimodal::VisionTensorLayout layout;
    layout.blocks.push_back(block);
    multimodal::SiglipBlockWeights weights;
    std::string error;
    CHECK(multimodal::load_siglip_block_weights(
        path.string(), index, config, layout, 0, &weights, &error));
    CHECK(error.empty());
    CHECK(weights.valid(hidden));
    CHECK(near(weights.query.weight.front(), 1.0f));
    CHECK(near(weights.attention_output.weight.back(), 4.0f));

    // Legacy raw ffn_down is hidden -> intermediate; raw ffn_up is intermediate -> hidden.
    CHECK(weights.feed_forward_input.input_size == hidden);
    CHECK(weights.feed_forward_input.output_size == intermediate);
    CHECK(near(weights.feed_forward_input.weight.front(), 5.0f));
    CHECK(near(weights.feed_forward_input.bias.front(), 0.5f));
    CHECK(weights.feed_forward_output.input_size == intermediate);
    CHECK(weights.feed_forward_output.output_size == hidden);
    CHECK(near(weights.feed_forward_output.weight.front(), 6.0f));
    CHECK(near(weights.feed_forward_output.bias.front(), 0.6f));

    CHECK(!multimodal::load_siglip_block_weights(
        path.string(), index, config, layout, 1, &weights, &error));
    CHECK(error.find("out of range") != std::string::npos);
    CHECK(!multimodal::load_siglip_block_weights(
        path.string(), index, config, layout, 0, nullptr, &error));
    std::filesystem::remove(path);
    return 0;
}
