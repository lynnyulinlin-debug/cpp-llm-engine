// Chapter 05：定义 Request phase、token 类型、生成限制和 logits callback。
// 契约：这些类型由 Request、Runtime、Scheduler 和 Sampler 共享。
// 验证：Chapter 05 测试检查 terminal state、context admission 和 callback 行为。
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
    Cancelled,
    TimedOut,
    Rejected,
    Failed,
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
    case RequestPhase::Cancelled:
        return "cancelled";
    case RequestPhase::TimedOut:
        return "timed_out";
    case RequestPhase::Rejected:
        return "rejected";
    case RequestPhase::Failed:
        return "failed";
    }

    return "unknown";
}

struct GenerationConfig {
    std::size_t max_new_tokens = 1;
    std::size_t max_context_tokens = 0;  // 0 means no tutorial admission limit.
    TokenId eos_token_id = -1;
    bool use_kv_cache = true;
};

}  // namespace chapter05_generation
