// Chapter 01：tokenizer 接口
// 职责：定义 prompt 到 token ids、token ids 到文本的最小接口。
// 契约：toy tokenizer 只用于教学契约，真实 BPE 不在此处手写。
// 验证：chapter01_tensor_loader_test.cpp 与 Chapter 00 llama.cpp 基线。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace chapter01 {

using TokenId = std::int32_t;
using TokenSequence = std::vector<TokenId>;

class Tokenizer {
public:
    explicit Tokenizer(std::size_t vocab_size = 64);

    std::size_t vocab_size() const;
    TokenSequence encode(std::string_view text) const;
    std::string decode(const TokenSequence& tokens) const;

private:
    std::size_t vocab_size_;
};

}  // namespace chapter01
