// Chapter 01：模型配置接口
// 职责：保存 tokenizer、Transformer 和 generation 所需的静态模型参数。
// 契约：valid() 是进入 Engine/Runtime 前的最小配置检查。
// 验证：chapter01_tensor_loader_test.cpp 与 Engine smoke test。
#pragma once

#include "tensor.h"
#include "model_metadata.h"
#include "tokenizer.h"

#include <cstddef>
#include <string>

namespace chapter01 {

struct ModelConfig {
    std::string architecture = "toy";
    std::size_t vocab_size = 64;
    std::size_t hidden_size = 16;
    std::size_t max_position = 32;
    std::size_t layer_count = 2;
    std::size_t max_new_tokens = 4;
    float rms_norm_epsilon = 1.0e-5f;
    TokenId bos_token_id = -1;
    TokenId eos_token_id = -1;

    static ModelConfig from_metadata(const ModelMetadata& metadata);
    bool valid() const;
    bool validate_tokenizer(const Tokenizer& tokenizer, std::string* error = nullptr) const;
    bool validate_tensor(const Tensor& tensor, std::string* error = nullptr) const;
};

}  // namespace chapter01
