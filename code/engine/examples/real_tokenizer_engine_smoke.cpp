// Engine：真实 tokenizer 接入 smoke test。
// 职责：使用 TinyLlama tokenizer 进入 Engine 的 prompt、Request 和 Runtime 链路。
// 限制：Runtime 仍是教学 Transformer，不代表真实 TinyLlama 权重 forward。
#include "cpp_llm_engine/engine.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_tokenizer_engine_smoke <model.gguf>\n";
        return 2;
    }

    std::string error;
    auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(argv[1], &error);
    if (tokenizer == nullptr) {
        std::cerr << "tokenizer load failed: " << error << '\n';
        return 1;
    }

    cpp_llm_engine::ModelConfig config;
    config.vocab_size = tokenizer->vocab_size();
    config.max_new_tokens = 2;
    const auto expected_prompt_tokens = tokenizer->encode("Hello TinyLlama");
    const cpp_llm_engine::Engine engine(config, std::move(tokenizer));
    const auto result = engine.generate("Hello TinyLlama");

    std::cout << "real_vocab_size: " << config.vocab_size << '\n'
              << "prompt_tokens: " << result.prompt_tokens.size() << '\n'
              << "expected_prompt_tokens: " << expected_prompt_tokens.size() << '\n'
              << "generated_tokens: " << result.generated_tokens.size() << '\n'
              << "finished: " << std::boolalpha << result.finished << '\n';
    return result.prompt_tokens == expected_prompt_tokens && result.finished ? 0 : 1;
}
