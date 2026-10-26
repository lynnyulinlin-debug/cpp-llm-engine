// Chapter 05：测试 scheduler ordering、terminal-request skipping 和 per-request generation accounting。
// 边界：覆盖 Serving Preview 入口，但不实现 production batch planner。
// 验证入口：Chapter 05 CTest target 或 scripts/run.sh。
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/model_executor.h"
#include "chapter05_generation/scheduler.h"
#include "chapter05_generation/sampler.h"
#include "chapter05_generation/transformer.h"

#include <cassert>
#include <vector>

namespace {

class FixedExecutor final : public chapter05_generation::RequestModelExecutor {
public:
    explicit FixedExecutor(chapter05_generation::TokenId token) : token_(token) {}

    bool prefill(const chapter05_generation::TokenSequence&,
                 chapter05_generation::Logits* logits,
                 std::string*) override {
        ++prefill_calls;
        *logits = make_logits();
        return true;
    }

    bool decode(chapter05_generation::TokenId,
                chapter05_generation::Logits* logits,
                std::string*) override {
        ++decode_calls;
        *logits = make_logits();
        return true;
    }

    std::size_t prefill_calls = 0;
    std::size_t decode_calls = 0;

private:
    chapter05_generation::Logits make_logits() const {
        chapter05_generation::Logits logits(8, 0.0f);
        logits[static_cast<std::size_t>(token_)] = 1.0f;
        return logits;
    }

    chapter05_generation::TokenId token_;
};

class FailingExecutor final : public chapter05_generation::RequestModelExecutor {
public:
    bool prefill(const chapter05_generation::TokenSequence&,
                 chapter05_generation::Logits*,
                 std::string* error) override {
        *error = "synthetic prefill failure";
        return false;
    }

    bool decode(chapter05_generation::TokenId,
                chapter05_generation::Logits*,
                std::string*) override {
        return false;
    }
};

}  // namespace

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
    chapter05_generation::Request cancelled({6});
    cancelled.cancel();

    std::vector<chapter05_generation::ScheduledRequest> requests;
    requests.push_back({0, &request_a, &sampler, {}});
    requests.push_back({1, &request_b, &sampler, {}});
    requests.push_back({2, &cancelled, &sampler, {}});

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

    assert(stats.scheduled_requests == 3);
    assert(stats.prefetched_requests == 2);
    assert(stats.generated_tokens == 4);
    assert(stats.completed_requests == 2);
    assert(request_a.finished());
    assert(request_b.finished());
    assert(request_a.generated_size() == 2);
    assert(request_b.generated_size() == 2);
    assert(request_a.cache().size() == 5);
    assert(request_b.cache().size() == 4);
    assert(cancelled.cache().empty());
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

    chapter05_generation::Runtime provider_runtime(config);
    chapter05_generation::Request provider_a({1, 2});
    chapter05_generation::Request provider_b({3, 4});
    auto executor_a = std::make_shared<FixedExecutor>(3);
    auto executor_b = std::make_shared<FixedExecutor>(4);
    const auto provider_stats = scheduler.run_with_stats(
        provider_runtime,
        {
            {10, &provider_a, &sampler, executor_a},
            {11, &provider_b, &sampler, executor_b},
        });
    assert(provider_stats.prefetched_requests == 2);
    assert(provider_stats.generated_tokens == 4);
    assert(provider_stats.completed_requests == 2);
    assert(provider_a.generated() == chapter05_generation::TokenSequence({3, 3}));
    assert(provider_b.generated() == chapter05_generation::TokenSequence({4, 4}));
    assert(executor_a->prefill_calls == 1 && executor_a->decode_calls == 1);
    assert(executor_b->prefill_calls == 1 && executor_b->decode_calls == 1);

    chapter05_generation::Request failed_request({1});
    FailingExecutor failing_executor;
    assert(provider_runtime.generate(failed_request, sampler, failing_executor) == 0);
    assert(failed_request.failed());
    assert(failed_request.failure_reason() == "synthetic prefill failure");
    assert(failed_request.generated().empty());

    chapter05_generation::Request invalid_decode_request({1});
    invalid_decode_request.mark_prefilled();
    FixedExecutor unused_executor(3);
    assert(provider_runtime.decode_one(
               invalid_decode_request, sampler, unused_executor) == 0);
    assert(invalid_decode_request.failed());
    assert(invalid_decode_request.failure_reason() ==
           "model decode requires a previously generated token");
    assert(unused_executor.decode_calls == 0);
    return 0;
}
