// Engine：Tokenizer backend 抽象。
// 职责：隔离 toy tokenizer 与真实 tokenizer，向 Engine 提供统一的 token contract。
// 契约：encode/decode 使用 Chapter 01 的 TokenSequence；backend 不负责模型推理。
#pragma once

#include "tokenizer.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace cpp_llm_engine {

class TokenizerBackend {
public:
    virtual ~TokenizerBackend() = default;

    virtual std::size_t vocab_size() const = 0;
    virtual chapter01::TokenSequence encode(std::string_view text) const = 0;
    virtual std::string decode(const chapter01::TokenSequence& tokens) const = 0;
};

}  // namespace cpp_llm_engine
