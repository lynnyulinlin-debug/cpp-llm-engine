// Chapter 01：Llama tensor schema 校验实现
// 职责：检查 GGUF tensor names、shape、层数和 Llama decoder layout。
// 契约：schema 只消费 index 和 metadata，不加载完整权重。
// 验证：chapter01_tensor_loader_test.cpp 与 gguf_inspect --validate-schema。
#include "model_schema.h"

#include <limits>
#include <sstream>
#include <unordered_map>

namespace chapter01 {
namespace {

struct ShapeDescriptor {
    std::string name;
    std::vector<std::size_t> shape;
    bool has_payload = true;
};

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

std::string layer_name(std::size_t index, const char* suffix) {
    return "blk." + std::to_string(index) + "." + suffix;
}

DecoderLayerLayout make_layer(std::size_t index) {
    return DecoderLayerLayout{
        index,
        layer_name(index, "attn_norm.weight"),
        layer_name(index, "attn_q.weight"),
        layer_name(index, "attn_k.weight"),
        layer_name(index, "attn_v.weight"),
        layer_name(index, "attn_output.weight"),
        layer_name(index, "ffn_norm.weight"),
        layer_name(index, "ffn_gate.weight"),
        layer_name(index, "ffn_up.weight"),
        layer_name(index, "ffn_down.weight"),
    };
}

bool checked_dimension(std::uint64_t value, std::size_t* result, std::string* error) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        return fail(error, "GGUF tensor dimension exceeds host size");
    }
    *result = static_cast<std::size_t>(value);
    return true;
}

bool validate_shapes(const ModelMetadata& metadata,
                     const std::vector<ShapeDescriptor>& descriptors,
                     LlamaTensorLayout* layout,
                     std::string* error) {
    if (layout == nullptr) return fail(error, "model layout output must not be null");
    if (metadata.architecture != "llama") {
        return fail(error, "strict tensor schema currently supports only llama architecture");
    }

    std::unordered_map<std::string, const ShapeDescriptor*> by_name;
    for (const auto& descriptor : descriptors) {
        if (!by_name.emplace(descriptor.name, &descriptor).second) {
            return fail(error, "duplicate tensor in model schema: " + descriptor.name);
        }
    }

    const auto find = [&](const std::string& name) -> const ShapeDescriptor* {
        const auto it = by_name.find(name);
        return it == by_name.end() ? nullptr : it->second;
    };
    const auto expect = [&](const std::string& name,
                            const std::vector<std::size_t>& shape) -> bool {
        const auto* descriptor = find(name);
        if (descriptor == nullptr) return fail(error, "missing Llama tensor: " + name);
        if (!descriptor->has_payload) return fail(error, "Llama tensor has no payload: " + name);
        if (descriptor->shape != shape) {
            std::ostringstream message;
            message << "shape mismatch for " << name;
            return fail(error, message.str());
        }
        return true;
    };

    if (!expect("token_embd.weight", {metadata.hidden_size, metadata.vocab_size}) ||
        !expect("output.weight", {metadata.hidden_size, metadata.vocab_size}) ||
        !expect("output_norm.weight", {metadata.hidden_size})) {
        return false;
    }

    LlamaTensorLayout built;
    built.embedding = "token_embd.weight";
    built.output = "output.weight";
    built.output_norm = "output_norm.weight";
    built.layers.reserve(metadata.layer_count);

    std::size_t intermediate_size = 0;
    std::size_t key_value_size = 0;
    for (std::size_t i = 0; i < metadata.layer_count; ++i) {
        const auto layer = make_layer(i);
        if (!expect(layer.attn_norm, {metadata.hidden_size}) ||
            !expect(layer.ffn_norm, {metadata.hidden_size}) ||
            !expect(layer.attn_q, {metadata.hidden_size, metadata.hidden_size}) ||
            !expect(layer.attn_output, {metadata.hidden_size, metadata.hidden_size})) {
            return false;
        }

        const auto* gate = find(layer.ffn_gate);
        const auto* up = find(layer.ffn_up);
        const auto* down = find(layer.ffn_down);
        const auto* key = find(layer.attn_k);
        const auto* value = find(layer.attn_v);
        if (gate == nullptr || up == nullptr || down == nullptr || key == nullptr || value == nullptr) {
            return fail(error, "Llama layer is missing FFN or KV projection tensor");
        }
        if (gate->shape.size() != 2 || up->shape != gate->shape ||
            down->shape.size() != 2 || down->shape[0] != gate->shape[1] ||
            down->shape[1] != metadata.hidden_size) {
            return fail(error, "FFN tensor shapes do not match Llama layer contract");
        }
        if (key->shape.size() != 2 || value->shape != key->shape ||
            key->shape[0] != metadata.hidden_size) {
            return fail(error, "attention K/V tensor shapes do not match Llama layer contract");
        }
        if (intermediate_size == 0) intermediate_size = gate->shape[1];
        if (key_value_size == 0) key_value_size = key->shape[1];
        if (intermediate_size != gate->shape[1] || key_value_size != key->shape[1]) {
            return fail(error, "Llama layer dimensions are inconsistent");
        }
        for (const auto* descriptor : {gate, up, down, key, value}) {
            if (!descriptor->has_payload) return fail(error, "Llama tensor has no payload: " + descriptor->name);
        }
        built.layers.push_back(layer);
    }
    built.intermediate_size = intermediate_size;
    built.key_value_size = key_value_size;
    *layout = std::move(built);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace

bool ModelSchema::validate_gguf(const ModelMetadata& metadata,
                                const GgufModelIndex& index,
                                LlamaTensorLayout* layout,
                                std::string* error) {
    std::vector<ShapeDescriptor> descriptors;
    descriptors.reserve(index.tensors.size());
    for (const auto& tensor : index.tensors) {
        ShapeDescriptor descriptor;
        descriptor.name = tensor.name;
        descriptor.shape.reserve(tensor.dimensions.size());
        for (const auto dimension : tensor.dimensions) {
            std::size_t converted = 0;
            if (!checked_dimension(dimension, &converted, error)) return false;
            descriptor.shape.push_back(converted);
        }
        descriptors.push_back(std::move(descriptor));
    }
    return validate_shapes(metadata, descriptors, layout, error);
}

bool ModelSchema::validate_tensors(const ModelMetadata& metadata,
                                   const std::unordered_map<std::string, Tensor>& tensors,
                                   LlamaTensorLayout* layout,
                                   std::string* error) {
    std::vector<ShapeDescriptor> descriptors;
    descriptors.reserve(tensors.size());
    for (const auto& [name, tensor] : tensors) {
        ShapeDescriptor descriptor;
        descriptor.name = name;
        descriptor.has_payload = tensor.bytes() > 0 && tensor.raw_data() != nullptr;
        descriptor.shape.reserve(tensor.shape().size());
        for (const auto dimension : tensor.shape()) {
            if (dimension < 0) return fail(error, "tensor shape contains a negative dimension: " + name);
            descriptor.shape.push_back(static_cast<std::size_t>(dimension));
        }
        descriptors.push_back(std::move(descriptor));
    }
    return validate_shapes(metadata, descriptors, layout, error);
}

}  // namespace chapter01
