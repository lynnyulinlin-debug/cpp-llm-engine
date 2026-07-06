#pragma once

#include "chapter05_generation/generation_types.h"

namespace chapter05_generation {

class KVCache {
public:
    void clear();
    void append(TokenId token);

    std::size_t size() const;
    const TokenSequence& tokens() const;
    bool empty() const;

private:
    TokenSequence tokens_;
};

}  // namespace chapter05_generation
