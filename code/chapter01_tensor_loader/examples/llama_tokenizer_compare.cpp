// Chapter 01：用真实 GGUF vocabulary 检查 tokenizer encode/decode。
// 职责：输出 TinyLlama 的 token ids 和 detokenized 文本，作为 llama.cpp reference 证据。
// 运行：llama_tokenizer_compare <model.gguf> <text> [--parse-special]。
#include "real_tokenizer.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4 ||
        (argc == 4 && std::string(argv[3]) != "--parse-special")) {
        std::cerr << "usage: llama_tokenizer_compare <model.gguf> <text> [--parse-special]\n";
        return 2;
    }
    std::string error;
    const auto tokenizer = chapter01::RealTokenizer::load(argv[1], &error);
    if (tokenizer == nullptr) {
        std::cerr << "load failed: " << error << "\n";
        return 1;
    }
    const bool parse_special = argc == 4;
    const auto tokens = tokenizer->encode(argv[2], false, parse_special);
    std::cout << "vocab_size=" << tokenizer->vocab_size() << "\n"
              << "token_count=" << tokens.size() << "\n"
              << "tokens=";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << tokens[index];
    }
    std::cout << "\ntext=" << tokenizer->decode(tokens, false, true) << "\n";
    return 0;
}
