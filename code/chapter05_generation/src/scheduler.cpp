// Chapter 05：实现 round-robin request progression 和 streaming callback dispatch。
// 契约：terminal、rejected request 被跳过，不触碰其 Runtime state。
// 验证：Scheduler 测试检查 fairness、completion accounting 和 request isolation。
#include "chapter05_generation/scheduler.h"

#include "chapter05_generation/request.h"
#include "chapter05_generation/model_executor.h"
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
        if (scheduled.executor == nullptr) {
            stats.prefetched_requests += runtime.prefill(*scheduled.request) > 0 ? 1 : 0;
        }
    }

    bool progress = true;
    while (progress) {
        progress = false;
        for (auto& scheduled : requests) {
            if (scheduled.request == nullptr || scheduled.sampler == nullptr || scheduled.request->terminal()) {
                continue;
            }

            const std::size_t before = scheduled.request->generated_size();
            const bool was_prefilled = scheduled.request->prefilled();
            const std::size_t decoded = scheduled.executor == nullptr
                ? runtime.decode_one(*scheduled.request, *scheduled.sampler)
                : runtime.decode_one(*scheduled.request, *scheduled.sampler,
                                     *scheduled.executor);
            if (decoded == 0) {
                continue;
            }

            if (!was_prefilled && scheduled.request->prefilled()) {
                ++stats.prefetched_requests;
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
