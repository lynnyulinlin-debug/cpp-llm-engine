// Chapter 01：Llama tensor schema 接口
// 职责：定义 embedding、norm、output 和 decoder layer tensor 的稳定映射。
// 契约：validate_gguf 成功后才能构造 DecoderInput。
// 验证：chapter01_tensor_loader_test.cpp。
#pragma once

#include "gguf_reader.h"
#include "model_metadata.h"
#include "tensor.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace chapter01 {

struct DecoderLayerLayout {
    std::size_t index = 0;
    std::string attn_norm;
    std::string attn_q;
    std::string attn_k;
    std::string attn_v;
    std::string attn_output;
    std::string ffn_norm;
    std::string ffn_gate;
    std::string ffn_up;
    std::string ffn_down;
};

struct LlamaTensorLayout {
    std::string embedding;
    std::string output;
    std::string output_norm;
    std::size_t intermediate_size = 0;
    std::size_t key_value_size = 0;
    std::vector<DecoderLayerLayout> layers;
};

class ModelSchema final {
public:
    static bool validate_gguf(const ModelMetadata& metadata,
                              const GgufModelIndex& index,
                              LlamaTensorLayout* layout,
                              std::string* error = nullptr);

    static bool validate_tensors(const ModelMetadata& metadata,
                                 const std::unordered_map<std::string, Tensor>& tensors,
                                 LlamaTensorLayout* layout,
                                 std::string* error = nullptr);
};

}  // namespace chapter01
