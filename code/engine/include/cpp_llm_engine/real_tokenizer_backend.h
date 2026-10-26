// Engine：真实 llama.cpp tokenizer backend 声明。
// 职责：把 Chapter 01 RealTokenizer 接入 Engine 的 TokenizerBackend contract。
// 契约：只替换文本到 token 的实现，不改变 Engine、Request 或 Runtime 状态。
#pragma once

#include "cpp_llm_engine/tokenizer_backend.h"

#include <memory>
#include <string>

namespace chapter01 {
class RealTokenizer;
}

namespace cpp_llm_engine {

class RealTokenizerBackend final : public TokenizerBackend {
public:
    static std::unique_ptr<RealTokenizerBackend> load(
        const std::string& model_path,
        std::string* error = nullptr);
    ~RealTokenizerBackend() override;

    std::size_t vocab_size() const override;
    chapter01::TokenSequence encode(std::string_view text) const override;
    std::string decode(const chapter01::TokenSequence& tokens) const override;

private:
    explicit RealTokenizerBackend(std::unique_ptr<chapter01::RealTokenizer> tokenizer);

    std::unique_ptr<chapter01::RealTokenizer> tokenizer_;
};

}  // namespace cpp_llm_engine
