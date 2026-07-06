#include "chapter05_generation/request.h"

#include <utility>

namespace chapter05_generation {

Request::Request(TokenSequence prompt)
    : prompt_(std::move(prompt)) {}

const TokenSequence& Request::prompt() const {
    return prompt_;
}

const TokenSequence& Request::generated() const {
    return generated_;
}

bool Request::finished() const {
    return phase_ == RequestPhase::Finished;
}

bool Request::prefilled() const {
    return phase_ != RequestPhase::Initial;
}

RequestPhase Request::phase() const {
    return phase_;
}

std::size_t Request::prompt_size() const {
    return prompt_.size();
}

std::size_t Request::generated_size() const {
    return generated_.size();
}

void Request::append_generated_token(TokenId token) {
    generated_.push_back(token);
}

void Request::mark_prefilled() {
    phase_ = RequestPhase::Prefilled;
}

void Request::mark_decoding() {
    phase_ = RequestPhase::Decoding;
}

void Request::mark_finished() {
    phase_ = RequestPhase::Finished;
}

}  // namespace chapter05_generation
