// Chapter 01：声明基于 llama.cpp vocabulary 的真实 tokenizer adapter。
// 职责：把真实 GGUF tokenizer 接到教程 TokenSequence，不复制 BPE/SentencePiece 算法。
// 验证：llama_tokenizer_compare 使用 TinyLlama GGUF 检查 encode/decode 和词表范围。
#pragma once

#include "tokenizer.h"

#include <memory>
#include <string>
#include <string_view>

struct llama_model;
struct llama_vocab;

namespace chapter01 {

class RealTokenizer final {
public:
    static std::unique_ptr<RealTokenizer> load(
        const std::string& model_path,
        std::string* error = nullptr);
    ~RealTokenizer();

    RealTokenizer(const RealTokenizer&) = delete;
    RealTokenizer& operator=(const RealTokenizer&) = delete;

    std::size_t vocab_size() const;
    TokenSequence encode(
        std::string_view text,
        bool add_special = false,
        bool parse_special = false) const;
    std::string decode(
        const TokenSequence& tokens,
        bool remove_special = false,
        bool unparse_special = true) const;

private:
    explicit RealTokenizer(llama_model* model);

    llama_model* model_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
};

}  // namespace chapter01
