// Chapter 01：声明真实 Llama decoder layer 权重装配器。
// 职责：按 ModelSchema 从 GGUF 按需加载一层权重，不复制整个模型到内存。
// 契约：只装配 tensor payload 和布局，不执行 Attention/FFN forward。
#pragma once

#include "gguf_reader.h"
#include "model_schema.h"
#include "tensor.h"

#include <cstddef>
#include <string>

namespace chapter01 {

struct RealLlamaLayerWeights {
    std::size_t index = 0;
    Tensor attn_norm;
    Tensor attn_q;
    Tensor attn_k;
    Tensor attn_v;
    Tensor attn_output;
    Tensor ffn_norm;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;

    bool valid() const;
};

class RealLlamaLayerLoader final {
public:
    static bool load(
        const std::string& model_path,
        const GgufModelIndex& index,
        const LlamaTensorLayout& layout,
        std::size_t layer_index,
        RealLlamaLayerWeights* output,
        std::string* error = nullptr);
};

}  // namespace chapter01
