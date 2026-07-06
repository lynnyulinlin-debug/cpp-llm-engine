#pragma once

#include "chapter05_generation/generation_types.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace chapter05_generation {

class Request;
class Runtime;
class Sampler;

struct ScheduledRequest {
    std::size_t request_id = 0;
    Request* request = nullptr;
    const Sampler* sampler = nullptr;
};

struct SchedulerStats {
    std::size_t scheduled_requests = 0;
    std::size_t prefetched_requests = 0;
    std::size_t generated_tokens = 0;
    std::size_t completed_requests = 0;
};

using TokenStreamCallback = std::function<void(std::size_t request_id, TokenId token, std::size_t step)>;

class RoundRobinScheduler final {
public:
    bool ready() const;
    SchedulerStats run_with_stats(
        Runtime& runtime,
        std::vector<ScheduledRequest> requests,
        const TokenStreamCallback& on_token = {}) const;
    std::size_t run(
        Runtime& runtime,
        std::vector<ScheduledRequest> requests,
        const TokenStreamCallback& on_token = {}) const;
};

}  // namespace chapter05_generation
