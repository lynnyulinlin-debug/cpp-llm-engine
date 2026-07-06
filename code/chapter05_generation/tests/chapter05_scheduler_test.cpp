#include "chapter05_generation/runtime.h"
#include "chapter05_generation/scheduler.h"
#include "chapter05_generation/sampler.h"
#include "chapter05_generation/transformer.h"

#include <cassert>
#include <vector>

int main() {
    chapter05_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.use_kv_cache = true;

    chapter05_generation::Embedding embedding(32);
    chapter05_generation::PositionEncoding position_encoding(16);
    chapter05_generation::Transformer transformer(2, embedding, position_encoding);
    chapter05_generation::Runtime runtime(config, transformer);
    chapter05_generation::GreedySampler sampler;

    chapter05_generation::Request request_a({1, 2, 3});
    chapter05_generation::Request request_b({4, 5});

    std::vector<chapter05_generation::ScheduledRequest> requests;
    requests.push_back({0, &request_a, &sampler});
    requests.push_back({1, &request_b, &sampler});

    std::vector<std::size_t> request_ids;
    std::vector<std::size_t> steps;
    std::vector<chapter05_generation::TokenId> tokens;
    const chapter05_generation::RoundRobinScheduler scheduler;
    const chapter05_generation::SchedulerStats stats = scheduler.run_with_stats(
        runtime,
        requests,
        [&](std::size_t request_id, chapter05_generation::TokenId token, std::size_t step) {
            request_ids.push_back(request_id);
            tokens.push_back(token);
            steps.push_back(step);
        });

    assert(stats.scheduled_requests == 2);
    assert(stats.prefetched_requests == 2);
    assert(stats.generated_tokens == 4);
    assert(stats.completed_requests == 2);
    assert(request_a.finished());
    assert(request_b.finished());
    assert(request_a.generated_size() == 2);
    assert(request_b.generated_size() == 2);
    assert(runtime.cache().size() == 9);
    assert(request_ids.size() == 4);
    assert(request_ids[0] == 0);
    assert(request_ids[1] == 1);
    assert(request_ids[2] == 0);
    assert(request_ids[3] == 1);
    assert(steps[0] == 0);
    assert(steps[1] == 0);
    assert(steps[2] == 1);
    assert(steps[3] == 1);
    assert(tokens[0] == 2);
    assert(tokens[1] == 2);
    assert(tokens[2] == 2);
    assert(tokens[3] == 2);
    return 0;
}
