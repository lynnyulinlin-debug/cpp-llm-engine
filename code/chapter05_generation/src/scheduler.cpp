#include "chapter05_generation/scheduler.h"

#include "chapter05_generation/request.h"
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/sampler.h"

namespace chapter05_generation {

bool RoundRobinScheduler::ready() const {
    return true;
}

SchedulerStats RoundRobinScheduler::run_with_stats(
    Runtime& runtime,
    std::vector<ScheduledRequest> requests,
    const TokenStreamCallback& on_token) const {
    SchedulerStats stats;
    stats.scheduled_requests = requests.size();

    if (!ready() || !runtime.ready()) {
        return stats;
    }

    for (auto& scheduled : requests) {
        if (scheduled.request == nullptr || scheduled.sampler == nullptr) {
            continue;
        }
        stats.prefetched_requests += runtime.prefill(*scheduled.request) > 0 ? 1 : 0;
    }

    bool progress = true;
    while (progress) {
        progress = false;
        for (auto& scheduled : requests) {
            if (scheduled.request == nullptr || scheduled.sampler == nullptr || scheduled.request->finished()) {
                continue;
            }

            const std::size_t before = scheduled.request->generated_size();
            const std::size_t decoded = runtime.decode_one(*scheduled.request, *scheduled.sampler);
            if (decoded == 0) {
                continue;
            }

            progress = true;
            stats.generated_tokens += decoded;
            if (on_token != nullptr && scheduled.request->generated_size() > before) {
                on_token(
                    scheduled.request_id,
                    scheduled.request->generated().back(),
                    scheduled.request->generated_size() - 1);
            }
        }
    }

    for (const auto& scheduled : requests) {
        if (scheduled.request != nullptr && scheduled.request->finished()) {
            ++stats.completed_requests;
        }
    }

    return stats;
}

std::size_t RoundRobinScheduler::run(
    Runtime& runtime,
    std::vector<ScheduledRequest> requests,
    const TokenStreamCallback& on_token) const {
    return run_with_stats(runtime, std::move(requests), on_token).generated_tokens;
}

}  // namespace chapter05_generation
