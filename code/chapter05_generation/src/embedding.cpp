#include "chapter05_generation/embedding.h"

namespace chapter05_generation {

Embedding::Embedding(std::size_t vocab_size)
    : vocab_size_(vocab_size) {}

std::size_t Embedding::vocab_size() const {
    return vocab_size_;
}

bool Embedding::ready() const {
    return vocab_size_ > 0;
}

Logits Embedding::forward(const TokenSequence& tokens) const {
    Logits hidden;
    hidden.reserve(tokens.size());
    for (TokenId token : tokens) {
        const float normalized = vocab_size_ > 0 ? static_cast<float>(token % static_cast<TokenId>(vocab_size_)) : static_cast<float>(token);
        hidden.push_back(normalized);
    }
    return hidden;
}

}  // namespace chapter05_generation
