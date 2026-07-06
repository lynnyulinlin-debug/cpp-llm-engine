#pragma once

#include <cstddef>
#include <functional>
#include <string_view>
#include <vector>

namespace chapter05_generation {

class Request;

using TokenId = int;
using TokenSequence = std::vector<TokenId>;
using Logits = std::vector<float>;
using LogitsProvider = std::function<Logits(const Request&, std::size_t)>;

enum class RequestPhase {
    Initial,
    Prefilled,
    Decoding,
    Finished,
};

inline std::string_view request_phase_name(RequestPhase phase) {
    switch (phase) {
    case RequestPhase::Initial:
        return "initial";
    case RequestPhase::Prefilled:
        return "prefilled";
    case RequestPhase::Decoding:
        return "decoding";
    case RequestPhase::Finished:
        return "finished";
    }

    return "unknown";
}

struct GenerationConfig {
    std::size_t max_new_tokens = 1;
    bool use_kv_cache = true;
};

}  // namespace chapter05_generation
