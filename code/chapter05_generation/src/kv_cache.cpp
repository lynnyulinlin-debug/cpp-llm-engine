#include "chapter05_generation/kv_cache.h"

namespace chapter05_generation {

void KVCache::clear() {
    tokens_.clear();
}

void KVCache::append(TokenId token) {
    tokens_.push_back(token);
}

std::size_t KVCache::size() const {
    return tokens_.size();
}

const TokenSequence& KVCache::tokens() const {
    return tokens_;
}

bool KVCache::empty() const {
    return tokens_.empty();
}

}  // namespace chapter05_generation
