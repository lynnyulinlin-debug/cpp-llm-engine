// Chapter 01：GGUF tokenizer metadata 接口
// 职责：定义真实 tokenizer 配置从 GGUF metadata 进入教程对象的结构。
// 契约：只保存词表和 special token metadata，不执行 BPE 分词。
// 验证：chapter01_tensor_loader_test.cpp。
#pragma once

#include "model_metadata_adapter.h"
#include "tokenizer.h"

#include <string>
#include <vector>

namespace chapter01 {

struct TokenizerMetadata {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<std::uint32_t> token_types;
    TokenId bos_token_id = -1;
    TokenId eos_token_id = -1;
    std::string chat_template;

    bool valid() const;
};

class TokenizerMetadataAdapter {
public:
    static bool from_gguf(const MetadataMap& values,
                          TokenizerMetadata* metadata,
                          std::string* error = nullptr);
};

}  // namespace chapter01
