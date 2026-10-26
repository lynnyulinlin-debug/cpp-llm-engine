// Engine：真实 tokenizer backend 实现。
// 职责：复用 Chapter 01 的 llama.cpp adapter，避免在 Engine 复制 tokenizer 算法。
// 验证：real_tokenizer_engine_smoke 使用 TinyLlama GGUF 运行 Engine 输入链路。
#include "cpp_llm_engine/real_tokenizer_backend.h"

#include "real_tokenizer.h"

#include <stdexcept>

namespace cpp_llm_engine {

std::unique_ptr<RealTokenizerBackend> RealTokenizerBackend::load(
    const std::string& model_path,
    std::string* error) {
    auto tokenizer = chapter01::RealTokenizer::load(model_path, error);
    if (tokenizer == nullptr) return nullptr;
    return std::unique_ptr<RealTokenizerBackend>(
        new RealTokenizerBackend(std::move(tokenizer)));
}

RealTokenizerBackend::RealTokenizerBackend(
    std::unique_ptr<chapter01::RealTokenizer> tokenizer)
    : tokenizer_(std::move(tokenizer)) {}

RealTokenizerBackend::~RealTokenizerBackend() = default;

std::size_t RealTokenizerBackend::vocab_size() const {
    return tokenizer_->vocab_size();
}

chapter01::TokenSequence RealTokenizerBackend::encode(std::string_view text) const {
    return tokenizer_->encode(text, false, false);
}

std::string RealTokenizerBackend::decode(
    const chapter01::TokenSequence& tokens) const {
    return tokenizer_->decode(tokens, false, true);
}

}  // namespace cpp_llm_engine
