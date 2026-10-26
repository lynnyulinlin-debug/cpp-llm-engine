// 多模态 CPU：视觉模型 GGUF schema 实现
// 职责：解析 IDEFICS3 视觉配置，并校验 SigLIP 与 projector tensor index。
// 契约：当前 schema 对应 SmolVLM 的 clip-vision/idefics3 mmproj 命名。
// 验证：multimodal_mmproj_schema_test 与 multimodal_mmproj_inspect。
#include "multimodal/mmproj_schema.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>

namespace multimodal {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

bool read_string(const chapter01::MetadataMap& metadata,
                 const std::string& key,
                 std::string* output,
                 std::string* error) {
    const auto iterator = metadata.find(key);
    if (iterator == metadata.end()) {
        set_error(error, "missing mmproj metadata: " + key);
        return false;
    }
    const auto* value = std::get_if<std::string>(&iterator->second);
    if (value == nullptr) {
        set_error(error, "mmproj metadata has wrong type: " + key);
        return false;
    }
    *output = *value;
    return true;
}

bool read_size(const chapter01::MetadataMap& metadata,
               const std::string& key,
               std::size_t* output,
               std::string* error) {
    const auto iterator = metadata.find(key);
    if (iterator == metadata.end()) {
        set_error(error, "missing mmproj metadata: " + key);
        return false;
    }

    std::uint64_t number = 0;
    if (const auto* value = std::get_if<std::uint64_t>(&iterator->second)) {
        number = *value;
    } else if (const auto* value = std::get_if<std::int64_t>(&iterator->second)) {
        if (*value < 0) {
            set_error(error, "mmproj metadata must be non-negative: " + key);
            return false;
        }
        number = static_cast<std::uint64_t>(*value);
    } else {
        set_error(error, "mmproj metadata has wrong type: " + key);
        return false;
    }
    if (number > std::numeric_limits<std::size_t>::max()) {
        set_error(error, "mmproj metadata exceeds size_t: " + key);
        return false;
    }
    *output = static_cast<std::size_t>(number);
    return true;
}

bool read_float(const chapter01::MetadataMap& metadata,
                const std::string& key,
                float* output,
                std::string* error) {
    const auto iterator = metadata.find(key);
    if (iterator == metadata.end()) {
        set_error(error, "missing mmproj metadata: " + key);
        return false;
    }
    const auto* value = std::get_if<float>(&iterator->second);
    if (value == nullptr || !std::isfinite(*value)) {
        set_error(error, "mmproj metadata must be a finite float: " + key);
        return false;
    }
    *output = *value;
    return true;
}

bool read_rgb_array(const chapter01::MetadataMap& metadata,
                    const std::string& key,
                    std::array<float, 3>* output,
                    std::string* error) {
    const auto iterator = metadata.find(key);
    if (iterator == metadata.end()) {
        set_error(error, "missing mmproj metadata: " + key);
        return false;
    }
    const auto* values = std::get_if<std::vector<float>>(&iterator->second);
    if (values == nullptr || values->size() != output->size()) {
        set_error(error, "mmproj metadata must contain three floats: " + key);
        return false;
    }
    for (std::size_t i = 0; i < output->size(); ++i) {
        if (!std::isfinite((*values)[i])) {
            set_error(error, "mmproj metadata contains non-finite value: " + key);
            return false;
        }
        (*output)[i] = (*values)[i];
    }
    return true;
}

bool checked_multiply(std::size_t lhs, std::size_t rhs, std::size_t* result) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t* result) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool validate_config(const VisionModelConfig& config, std::string* error) {
    if (config.architecture != "clip") {
        set_error(error, "unsupported mmproj architecture: " + config.architecture);
        return false;
    }
    if (config.projector_type != "idefics3") {
        set_error(error, "unsupported mmproj projector: " + config.projector_type);
        return false;
    }
    if (config.image_size == 0 || config.patch_size == 0 || config.hidden_size == 0 ||
        config.intermediate_size == 0 || config.block_count == 0 ||
        config.attention_head_count == 0 || config.projection_dim == 0 ||
        config.merge_size == 0) {
        set_error(error, "mmproj dimensions must be non-zero");
        return false;
    }
    if (config.image_size % config.patch_size != 0) {
        set_error(error, "image size must be divisible by patch size");
        return false;
    }
    const std::size_t patches_per_side = config.image_size / config.patch_size;
    if (patches_per_side % config.merge_size != 0) {
        set_error(error, "patch grid must be divisible by projector scale factor");
        return false;
    }
    if (config.hidden_size % config.attention_head_count != 0) {
        set_error(error, "vision hidden size must be divisible by attention head count");
        return false;
    }
    if (!std::isfinite(config.layer_norm_epsilon) || !(config.layer_norm_epsilon > 0.0f)) {
        set_error(error, "layer norm epsilon must be positive");
        return false;
    }
    for (float value : config.image_mean) {
        if (!std::isfinite(value)) {
            set_error(error, "image mean must contain finite values");
            return false;
        }
    }
    for (float value : config.image_std) {
        if (!std::isfinite(value) || !(value > 0.0f)) {
            set_error(error, "image standard deviation must be positive");
            return false;
        }
    }
    return true;
}

VisionBlockLayout make_block_layout(std::size_t index) {
    const std::string prefix = "v.blk." + std::to_string(index) + ".";
    return VisionBlockLayout{
        index,
        prefix + "ln1.weight",
        prefix + "ln1.bias",
        prefix + "ln2.weight",
        prefix + "ln2.bias",
        prefix + "attn_q.weight",
        prefix + "attn_q.bias",
        prefix + "attn_k.weight",
        prefix + "attn_k.bias",
        prefix + "attn_v.weight",
        prefix + "attn_v.bias",
        prefix + "attn_out.weight",
        prefix + "attn_out.bias",
        prefix + "ffn_up.weight",
        prefix + "ffn_up.bias",
        prefix + "ffn_down.weight",
        prefix + "ffn_down.bias"};
}

bool expect_tensor(const std::unordered_map<std::string, const chapter01::GgufTensorInfo*>& tensors,
                   const std::string& name,
                   const std::vector<std::uint64_t>& dimensions,
                   std::string* error) {
    const auto iterator = tensors.find(name);
    if (iterator == tensors.end()) {
        set_error(error, "missing mmproj tensor: " + name);
        return false;
    }
    if (iterator->second->dimensions != dimensions) {
        set_error(error, "mmproj tensor shape mismatch: " + name);
        return false;
    }
    return true;
}

}  // namespace

bool MmprojSchema::config_from_gguf(const chapter01::MetadataMap& metadata,
                                    VisionModelConfig* config,
                                    std::string* error) {
    if (config == nullptr) {
        set_error(error, "vision model config output must not be null");
        return false;
    }

    VisionModelConfig parsed;
    if (!read_string(metadata, "general.architecture", &parsed.architecture, error) ||
        !read_string(metadata, "clip.projector_type", &parsed.projector_type, error) ||
        !read_size(metadata, "clip.vision.image_size", &parsed.image_size, error) ||
        !read_size(metadata, "clip.vision.patch_size", &parsed.patch_size, error) ||
        !read_size(metadata, "clip.vision.embedding_length", &parsed.hidden_size, error) ||
        !read_size(metadata, "clip.vision.feed_forward_length", &parsed.intermediate_size, error) ||
        !read_size(metadata, "clip.vision.block_count", &parsed.block_count, error) ||
        !read_size(metadata, "clip.vision.attention.head_count", &parsed.attention_head_count, error) ||
        !read_size(metadata, "clip.vision.projection_dim", &parsed.projection_dim, error) ||
        !read_size(metadata, "clip.vision.projector.scale_factor", &parsed.merge_size, error) ||
        !read_float(metadata,
                    "clip.vision.attention.layer_norm_epsilon",
                    &parsed.layer_norm_epsilon,
                    error) ||
        !read_rgb_array(metadata, "clip.vision.image_mean", &parsed.image_mean, error) ||
        !read_rgb_array(metadata, "clip.vision.image_std", &parsed.image_std, error)) {
        return false;
    }

    if (!validate_config(parsed, error)) return false;

    *config = std::move(parsed);
    if (error != nullptr) error->clear();
    return true;
}

bool MmprojSchema::validate_gguf(const VisionModelConfig& config,
                                 const chapter01::GgufModelIndex& index,
                                 VisionTensorLayout* layout,
                                 std::string* error) {
    if (layout == nullptr) {
        set_error(error, "vision tensor layout output must not be null");
        return false;
    }
    if (!validate_config(config, error)) return false;

    std::unordered_map<std::string, const chapter01::GgufTensorInfo*> tensors;
    tensors.reserve(index.tensors.size());
    for (const auto& tensor : index.tensors) {
        if (!tensors.emplace(tensor.name, &tensor).second) {
            set_error(error, "duplicate mmproj tensor: " + tensor.name);
            return false;
        }
    }

    VisionTensorLayout parsed;
    parsed.patch_embedding_weight = "v.patch_embd.weight";
    parsed.patch_embedding_bias = "v.patch_embd.bias";
    parsed.position_embedding_weight = "v.position_embd.weight";
    parsed.post_layer_norm_weight = "v.post_ln.weight";
    parsed.post_layer_norm_bias = "v.post_ln.bias";
    parsed.projector_weight = "mm.model.fc.weight";
    std::size_t block_tensor_count = 0;
    if (!checked_multiply(16, config.block_count, &block_tensor_count) ||
        !checked_add(6, block_tensor_count, &parsed.expected_tensor_count)) {
        set_error(error, "mmproj tensor count calculation overflow");
        return false;
    }
    if (index.tensors.size() != parsed.expected_tensor_count) {
        set_error(error,
                  "mmproj tensor count mismatch: expected " +
                      std::to_string(parsed.expected_tensor_count) + ", got " +
                      std::to_string(index.tensors.size()));
        return false;
    }

    std::size_t position_count = 0;
    std::size_t merged_hidden = 0;
    const std::size_t patches_per_side = config.image_size / config.patch_size;
    if (!checked_multiply(patches_per_side, patches_per_side, &position_count) ||
        !checked_multiply(config.hidden_size, config.merge_size, &merged_hidden) ||
        !checked_multiply(merged_hidden, config.merge_size, &merged_hidden)) {
        set_error(error, "mmproj tensor shape calculation overflow");
        return false;
    }

    const auto dim = [](std::size_t value) { return static_cast<std::uint64_t>(value); };
    if (!expect_tensor(tensors,
                       parsed.patch_embedding_weight,
                       {dim(config.patch_size), dim(config.patch_size), 3, dim(config.hidden_size)},
                       error) ||
        !expect_tensor(tensors, parsed.patch_embedding_bias, {dim(config.hidden_size)}, error) ||
        !expect_tensor(tensors,
                       parsed.position_embedding_weight,
                       {dim(config.hidden_size), dim(position_count)},
                       error) ||
        !expect_tensor(tensors, parsed.post_layer_norm_weight, {dim(config.hidden_size)}, error) ||
        !expect_tensor(tensors, parsed.post_layer_norm_bias, {dim(config.hidden_size)}, error) ||
        !expect_tensor(tensors,
                       parsed.projector_weight,
                       {dim(merged_hidden), dim(config.projection_dim)},
                       error)) {
        return false;
    }

    parsed.blocks.reserve(config.block_count);
    for (std::size_t block_index = 0; block_index < config.block_count; ++block_index) {
        VisionBlockLayout block = make_block_layout(block_index);
        const std::vector<std::uint64_t> hidden{dim(config.hidden_size)};
        const std::vector<std::uint64_t> attention{dim(config.hidden_size), dim(config.hidden_size)};
        if (!expect_tensor(tensors, block.layer_norm_1_weight, hidden, error) ||
            !expect_tensor(tensors, block.layer_norm_1_bias, hidden, error) ||
            !expect_tensor(tensors, block.layer_norm_2_weight, hidden, error) ||
            !expect_tensor(tensors, block.layer_norm_2_bias, hidden, error) ||
            !expect_tensor(tensors, block.attention_query_weight, attention, error) ||
            !expect_tensor(tensors, block.attention_query_bias, hidden, error) ||
            !expect_tensor(tensors, block.attention_key_weight, attention, error) ||
            !expect_tensor(tensors, block.attention_key_bias, hidden, error) ||
            !expect_tensor(tensors, block.attention_value_weight, attention, error) ||
            !expect_tensor(tensors, block.attention_value_bias, hidden, error) ||
            !expect_tensor(tensors, block.attention_output_weight, attention, error) ||
            !expect_tensor(tensors, block.attention_output_bias, hidden, error) ||
            !expect_tensor(tensors,
                           block.feed_forward_up_weight,
                           {dim(config.intermediate_size), dim(config.hidden_size)},
                           error) ||
            !expect_tensor(tensors,
                           block.feed_forward_up_bias,
                           {dim(config.hidden_size)},
                           error) ||
            !expect_tensor(tensors,
                           block.feed_forward_down_weight,
                           {dim(config.hidden_size), dim(config.intermediate_size)},
                           error) ||
            !expect_tensor(tensors,
                           block.feed_forward_down_bias,
                           {dim(config.intermediate_size)},
                           error)) {
            return false;
        }
        parsed.blocks.push_back(std::move(block));
    }

    *layout = std::move(parsed);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace multimodal
