#pragma once

#include "chapter05_generation/generation_types.h"

#include <cstddef>

namespace chapter05_generation {

class Embedding {
public:
    Embedding(std::size_t vocab_size = 0);

    std::size_t vocab_size() const;
    bool ready() const;
    Logits forward(const TokenSequence& tokens) const;

private:
    std::size_t vocab_size_ = 0;
};

}  // namespace chapter05_generation
