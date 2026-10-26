// Chapter 01：tokenizer 教学实现
// 职责：提供早期契约测试使用的 toy token encode/decode 映射。
// 契约：不冒充真实 BPE；真实模型 tokenizer metadata 由 tokenizer_metadata 读取。
// 验证：chapter01_tensor_loader_test.cpp。
#include "tokenizer.h"

#include <cctype>
#include <stdexcept>

namespace chapter01 {

Tokenizer::Tokenizer(std::size_t vocab_size)
    : vocab_size_(vocab_size) {
    if (vocab_size_ < 27) {
        throw std::invalid_argument("tokenizer vocabulary must contain at least 27 entries");
    }
}

std::size_t Tokenizer::vocab_size() const {
    return vocab_size_;
}

TokenSequence Tokenizer::encode(std::string_view text) const {
    TokenSequence tokens;
    tokens.reserve(text.size());
    for (const unsigned char character : text) {
        if (std::isalpha(character)) {
            const TokenId token = static_cast<TokenId>(
                std::tolower(character) - 'a' + 1);
            tokens.push_back(token < static_cast<TokenId>(vocab_size_) ? token : 0);
        } else {
            tokens.push_back(0);
        }
    }

    if (tokens.empty()) {
        tokens.push_back(0);
    }
    return tokens;
}

std::string Tokenizer::decode(const TokenSequence& tokens) const {
    std::string text;
    text.reserve(tokens.size());
    for (const TokenId token : tokens) {
        text.push_back(token <= 0 ? 'a' : static_cast<char>('a' + ((token - 1) % 26)));
    }
    return text;
}

}  // namespace chapter01
