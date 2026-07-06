#pragma once

#include "chapter05_generation/generation_types.h"

namespace chapter05_generation {

class Request {
public:
    Request() = default;
    explicit Request(TokenSequence prompt);

    const TokenSequence& prompt() const;
    const TokenSequence& generated() const;
    bool finished() const;
    bool prefilled() const;
    RequestPhase phase() const;
    std::size_t prompt_size() const;
    std::size_t generated_size() const;

    void append_generated_token(TokenId token);
    void mark_prefilled();
    void mark_decoding();
    void mark_finished();

private:
    TokenSequence prompt_;
    TokenSequence generated_;
    RequestPhase phase_ = RequestPhase::Initial;
};

}  // namespace chapter05_generation
