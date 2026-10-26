// Chapter 01：实现基于 llama.cpp vocabulary 的真实 tokenizer adapter。
// 契约：只负责 tokenize/detokenize 和 vocab 生命周期，模型推理仍由调用方负责。
// 验证：llama_tokenizer_compare 加载真实 TinyLlama GGUF 并打印 token 对照。
#include "real_tokenizer.h"

#include <llama.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace chapter01 {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

}  // namespace

std::unique_ptr<RealTokenizer> RealTokenizer::load(
    const std::string& model_path,
    std::string* error) {
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), params);
    if (model == nullptr) {
        llama_backend_free();
        set_error(error, "failed to load GGUF vocabulary: " + model_path);
        return nullptr;
    }
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealTokenizer>(new RealTokenizer(model));
}

RealTokenizer::RealTokenizer(llama_model* model)
    : model_(model), vocab_(llama_model_get_vocab(model)) {}

RealTokenizer::~RealTokenizer() {
    llama_model_free(model_);
    llama_backend_free();
}

std::size_t RealTokenizer::vocab_size() const {
    return static_cast<std::size_t>(llama_vocab_n_tokens(vocab_));
}

TokenSequence RealTokenizer::encode(
    std::string_view text,
    bool add_special,
    bool parse_special) const {
    if (text.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("tokenizer input is too large");
    }
    const int32_t required = llama_tokenize(
        vocab_, text.data(), static_cast<int32_t>(text.size()), nullptr, 0,
        add_special, parse_special);
    if (required == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("llama tokenizer input size overflow");
    }
    const int32_t count = required < 0 ? -required : required;
    TokenSequence tokens(static_cast<std::size_t>(count));
    if (count == 0) return tokens;
    const int32_t written = llama_tokenize(
        vocab_, text.data(), static_cast<int32_t>(text.size()),
        reinterpret_cast<llama_token*>(tokens.data()), count,
        add_special, parse_special);
    if (written < 0) throw std::runtime_error("llama tokenizer failed to encode input");
    tokens.resize(static_cast<std::size_t>(written));
    return tokens;
}

std::string RealTokenizer::decode(
    const TokenSequence& tokens,
    bool remove_special,
    bool unparse_special) const {
    if (tokens.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("token sequence is too large");
    }
    const auto* llama_tokens = reinterpret_cast<const llama_token*>(tokens.data());
    const int32_t required = llama_detokenize(
        vocab_, llama_tokens, static_cast<int32_t>(tokens.size()), nullptr, 0,
        remove_special, unparse_special);
    if (required < 0 && required == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("llama detokenizer output size overflow");
    }
    const int32_t count = required < 0 ? -required : required;
    std::string text(static_cast<std::size_t>(count), '\0');
    if (count == 0) return text;
    const int32_t written = llama_detokenize(
        vocab_, llama_tokens, static_cast<int32_t>(tokens.size()),
        text.data(), count, remove_special, unparse_special);
    if (written < 0) throw std::runtime_error("llama tokenizer failed to decode tokens");
    text.resize(static_cast<std::size_t>(written));
    return text;
}

}  // namespace chapter01
