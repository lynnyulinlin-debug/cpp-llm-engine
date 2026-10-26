// Chapter 05：测试 token-level step planning、动态 batch 和 block addressing。
// 边界：验证逻辑选择与推进，不测试物理 KV allocation 或 PagedAttention kernel。
// 验证入口：Chapter 05 CTest target 或 scripts/run.sh。
#include "chapter05_generation/serving_preview.h"

#include "chapter05_generation/model_executor.h"
#include "chapter05_generation/request.h"
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/sampler.h"

#include <cassert>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class FixedTokenExecutor final : public chapter05_generation::RequestModelExecutor {
public:
    explicit FixedTokenExecutor(chapter05_generation::TokenId token) : token_(token) {}

    bool prefill(const chapter05_generation::TokenSequence&,
                 chapter05_generation::Logits* logits,
                 std::string*) override {
        ++prefill_calls;
        return write(logits);
    }

    bool decode(chapter05_generation::TokenId,
                chapter05_generation::Logits* logits,
                std::string*) override {
        ++decode_calls;
        return write(logits);
    }

    std::size_t prefill_calls = 0;
    std::size_t decode_calls = 0;

private:
    bool write(chapter05_generation::Logits* logits) const {
        logits->assign(8, 0.0f);
        (*logits)[static_cast<std::size_t>(token_)] = 1.0f;
        return true;
    }

    chapter05_generation::TokenId token_;
};

class ChunkedTokenExecutor final : public chapter05_generation::RequestModelExecutor {
public:
    explicit ChunkedTokenExecutor(
        chapter05_generation::TokenId token,
        bool fail_first_chunk = false)
        : token_(token), fail_first_chunk_(fail_first_chunk) {}

    bool prefill(const chapter05_generation::TokenSequence& prompt,
                 chapter05_generation::Logits* logits,
                 std::string* error) override {
        return prefill_chunk(prompt, true, true, logits, error);
    }

    bool supports_chunked_prefill() const override { return true; }

    bool prefill_chunk(const chapter05_generation::TokenSequence& prompt,
                       bool start_new_sequence,
                       bool produce_logits,
                       chapter05_generation::Logits* logits,
                       std::string* error) override {
        chunk_sizes.push_back(prompt.size());
        starts.push_back(start_new_sequence);
        finals.push_back(produce_logits);
        if (fail_first_chunk_ && chunk_sizes.size() == 1) {
            if (error != nullptr) *error = "synthetic chunk failure";
            return false;
        }
        if (produce_logits) {
            logits->assign(8, 0.0f);
            (*logits)[static_cast<std::size_t>(token_)] = 1.0f;
        } else {
            assert(logits == nullptr);
        }
        return true;
    }

    bool decode(chapter05_generation::TokenId,
                chapter05_generation::Logits* logits,
                std::string*) override {
        logits->assign(8, 0.0f);
        (*logits)[static_cast<std::size_t>(token_)] = 1.0f;
        return true;
    }

    std::vector<std::size_t> chunk_sizes;
    std::vector<bool> starts;
    std::vector<bool> finals;

private:
    chapter05_generation::TokenId token_;
    bool fail_first_chunk_ = false;
};

}  // namespace

int main() {
    chapter05_generation::Request request_a({1, 2});
    chapter05_generation::Request request_b({3, 4, 5});
    chapter05_generation::Request cancelled({4});
    cancelled.cancel();
    chapter05_generation::GreedySampler sampler;

    auto executor_a = std::make_shared<FixedTokenExecutor>(5);
    auto executor_b = std::make_shared<FixedTokenExecutor>(6);
    std::vector<chapter05_generation::ScheduledRequest> requests = {
        {7, &request_a, &sampler, executor_a},
        {8, &request_b, &sampler, executor_b},
        {9, &cancelled, &sampler, {}},
    };

    const chapter05_generation::StepBatchPlanner planner;
    const auto first_plan = planner.build(requests, 2);
    assert(first_plan.candidates.size() == 1);
    assert(first_plan.scheduled_tokens == 2);
    assert(first_plan.deferred_requests == 1);
    assert(first_plan.candidates[0].request_id == 7);
    assert(first_plan.candidates[0].prefill);

    chapter05_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 16;
    chapter05_generation::Runtime runtime(config);
    const chapter05_generation::ContinuousBatchScheduler scheduler;
    const auto first_step = scheduler.run_step(runtime, requests, 2);
    assert(first_step.prefill_requests == 1);
    assert(first_step.generated_tokens == 1);
    assert(first_step.deferred_requests == 1);
    assert(request_a.generated() == chapter05_generation::TokenSequence({5}));
    assert(!request_a.terminal());
    assert(!request_b.prefilled());

    chapter05_generation::Request request_c({4, 5});
    auto executor_c = std::make_shared<FixedTokenExecutor>(7);
    requests.push_back({10, &request_c, &sampler, executor_c});

    const auto mixed_plan = planner.build(requests, 4);
    assert(mixed_plan.candidates.size() == 2);
    assert(mixed_plan.decode_requests == 1);
    assert(mixed_plan.prefill_requests == 1);
    assert(mixed_plan.candidates[0].request_id == 7);
    assert(!mixed_plan.candidates[0].prefill);
    assert(mixed_plan.candidates[1].request_id == 8);
    assert(mixed_plan.candidates[1].prefill);
    assert(mixed_plan.scheduled_tokens == 4);

    const auto mixed_step = scheduler.run_step(runtime, requests, 4);
    assert(mixed_step.generated_tokens == 2);
    assert(request_a.finished());
    assert(request_b.generated() == chapter05_generation::TokenSequence({6}));
    assert(!request_c.prefilled());

    const auto dynamic_step = scheduler.run_step(runtime, requests, 3);
    assert(dynamic_step.decode_requests == 1);
    assert(dynamic_step.prefill_requests == 1);
    assert(dynamic_step.generated_tokens == 2);
    assert(request_b.finished());
    assert(request_c.generated() == chapter05_generation::TokenSequence({7}));
    const auto final_step = scheduler.run_step(runtime, requests, 1);
    assert(final_step.generated_tokens == 1);
    assert(request_c.finished());
    assert(executor_a->prefill_calls == 1 && executor_a->decode_calls == 1);
    assert(executor_b->prefill_calls == 1 && executor_b->decode_calls == 1);
    assert(executor_c->prefill_calls == 1 && executor_c->decode_calls == 1);
    assert(scheduler.run_step(runtime, requests, 4).scheduled_requests == 0);

    bool invalid_budget = false;
    try {
        (void)planner.build(requests, 0);
    } catch (const std::invalid_argument&) {
        invalid_budget = true;
    }
    assert(invalid_budget);

    bool rejected = false;
    try {
        chapter05_generation::BlockTable invalid(0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    bool invalid_pool = false;
    try {
        chapter05_generation::KvBlockPool zero_pool(0, 2);
    } catch (const std::invalid_argument&) {
        invalid_pool = true;
    }
    assert(invalid_pool);
    bool overflow_pool = false;
    try {
        chapter05_generation::KvBlockPool overflow(
            2, std::numeric_limits<std::size_t>::max());
    } catch (const std::overflow_error&) {
        overflow_pool = true;
    }
    assert(overflow_pool);

    chapter05_generation::KvBlockPool pool(3, 2);
    assert(pool.total_blocks() == 3 && pool.free_blocks() == 3);
    assert(pool.reserve(20, 3));
    assert(pool.block_count(20) == 2);
    assert(pool.commit_tokens(20, 3));
    assert(pool.locate(20, 0).block == 0 && pool.locate(20, 0).slot == 0);
    assert(pool.locate(20, 2).block == 1 && pool.locate(20, 2).slot == 0);
    bool locate_rejected = false;
    try {
        (void)pool.locate(20, 3);
    } catch (const std::out_of_range&) {
        locate_rejected = true;
    }
    assert(locate_rejected);
    assert(!pool.reserve(21, 3));
    assert(!pool.contains(21));
    assert(pool.free_blocks() == 1);
    assert(pool.reserve(21, 2));
    assert(pool.commit_tokens(21, 2));
    assert(pool.locate(21, 1).block == 2 && pool.locate(21, 1).slot == 1);
    assert(pool.release(20));
    assert(pool.free_blocks() == 2);
    assert(pool.reserve(21, 3));
    assert(pool.block_count(21) == 2);
    assert(pool.trim_to_tokens(21, 2));
    assert(pool.block_count(21) == 1);
    assert(pool.release(21));
    assert(pool.free_blocks() == 3);
    assert(!pool.release(99));

    chapter05_generation::GenerationConfig block_config;
    block_config.max_new_tokens = 1;
    block_config.max_context_tokens = 16;
    chapter05_generation::Runtime block_runtime(block_config);
    chapter05_generation::Request block_request_a({1, 2, 3});
    chapter05_generation::Request block_request_b({4, 5});
    auto block_executor_a = std::make_shared<FixedTokenExecutor>(5);
    auto block_executor_b = std::make_shared<FixedTokenExecutor>(6);
    const std::vector<chapter05_generation::ScheduledRequest> block_requests = {
        {30, &block_request_a, &sampler, block_executor_a},
        {31, &block_request_b, &sampler, block_executor_b},
    };
    chapter05_generation::KvBlockPool scheduling_pool(2, 2);
    const auto block_limited = scheduler.run_step(
        block_runtime, block_requests, 5, 0, {}, &scheduling_pool);
    assert(block_limited.scheduled_requests == 1);
    assert(block_limited.scheduled_tokens == 3);
    assert(block_limited.deferred_requests == 1);
    assert(block_request_a.finished());
    assert(!block_request_b.prefilled());
    assert(block_executor_b->prefill_calls == 0);
    assert(scheduling_pool.used_blocks() == 2);
    assert(scheduling_pool.release(30));
    const auto after_release = scheduler.run_step(
        block_runtime, block_requests, 2, 0, {}, &scheduling_pool);
    assert(after_release.scheduled_requests == 1);
    assert(block_request_b.finished());
    assert(scheduling_pool.locate(31, 0).block == 0);
    assert(scheduling_pool.release(31));
    assert(scheduling_pool.free_blocks() == 2);

    chapter05_generation::Request chunk_a({1, 2, 3, 4, 5});
    chapter05_generation::Request chunk_b({6, 7, 1, 2, 3});
    auto chunk_executor_a = std::make_shared<ChunkedTokenExecutor>(4);
    auto chunk_executor_b = std::make_shared<ChunkedTokenExecutor>(5);
    const std::vector<chapter05_generation::ScheduledRequest> chunk_requests = {
        {40, &chunk_a, &sampler, chunk_executor_a},
        {41, &chunk_b, &sampler, chunk_executor_b},
    };
    chapter05_generation::KvBlockPool chunk_pool(6, 2);
    const auto chunk_step_a = scheduler.run_step(
        block_runtime, chunk_requests, 2, 0, {}, &chunk_pool);
    assert(chunk_step_a.scheduled_tokens == 2 && chunk_step_a.generated_tokens == 0);
    assert(chunk_a.prefill_offset() == 2 && chunk_b.prefill_offset() == 0);
    assert(chunk_pool.token_count(40) == 2);

    const auto chunk_step_b = scheduler.run_step(
        block_runtime, chunk_requests, 2, 0, {}, &chunk_pool);
    assert(chunk_step_b.generated_tokens == 0);
    assert(chunk_a.prefill_offset() == 2 && chunk_b.prefill_offset() == 2);
    assert(chunk_pool.token_count(41) == 2);

    (void)scheduler.run_step(block_runtime, chunk_requests, 2, 0, {}, &chunk_pool);
    (void)scheduler.run_step(block_runtime, chunk_requests, 2, 0, {}, &chunk_pool);
    const auto chunk_finish_a = scheduler.run_step(
        block_runtime, chunk_requests, 2, 0, {}, &chunk_pool);
    assert(chunk_finish_a.scheduled_tokens == 2 && chunk_finish_a.generated_tokens == 2);
    assert(chunk_a.finished() && chunk_b.finished());
    const auto chunk_finish_b = scheduler.run_step(
        block_runtime, chunk_requests, 2, 0, {}, &chunk_pool);
    assert(chunk_finish_b.scheduled_requests == 0 && chunk_finish_b.generated_tokens == 0);
    assert(chunk_executor_a->chunk_sizes == std::vector<std::size_t>({2, 2, 1}));
    assert(chunk_executor_a->starts == std::vector<bool>({true, false, false}));
    assert(chunk_executor_a->finals == std::vector<bool>({false, false, true}));
    assert(chunk_executor_b->chunk_sizes == std::vector<std::size_t>({2, 2, 1}));
    assert(chunk_pool.token_count(40) == 5 && chunk_pool.token_count(41) == 5);

    chapter05_generation::Request failed_chunk({1, 2, 3, 4});
    auto failing_chunk_executor = std::make_shared<ChunkedTokenExecutor>(6, true);
    chapter05_generation::KvBlockPool rollback_pool(2, 2);
    const auto failed_step = scheduler.run_step(
        block_runtime,
        {{50, &failed_chunk, &sampler, failing_chunk_executor}},
        2,
        0,
        {},
        &rollback_pool);
    assert(failed_step.scheduled_requests == 1 && failed_step.generated_tokens == 0);
    assert(failed_chunk.failed() && failed_chunk.prefill_offset() == 0);
    assert(!rollback_pool.contains(50) && rollback_pool.free_blocks() == 2);

    chapter05_generation::GenerationConfig fairness_config;
    fairness_config.max_new_tokens = 3;
    chapter05_generation::Runtime fairness_runtime(fairness_config);
    chapter05_generation::Request decoding_request({1});
    auto decoding_executor = std::make_shared<FixedTokenExecutor>(3);
    assert(fairness_runtime.decode_one(
        decoding_request, sampler, *decoding_executor) == 1);
    chapter05_generation::Request waiting_prefill({2, 3, 4, 5});
    auto waiting_executor = std::make_shared<ChunkedTokenExecutor>(4);
    const std::size_t generated_before = decoding_request.generated_size();
    const auto starvation_guard = scheduler.run_step(
        fairness_runtime,
        {
            {60, &decoding_request, &sampler, decoding_executor},
            {61, &waiting_prefill, &sampler, waiting_executor},
        },
        1,
        1);
    assert(starvation_guard.prefill_requests == 1);
    assert(starvation_guard.decode_requests == 0);
    assert(decoding_request.generated_size() == generated_before);
    assert(waiting_prefill.prefill_offset() == 1);

    bool progress_rejected = false;
    try {
        chapter05_generation::Request invalid_progress({1});
        invalid_progress.advance_prefill(2);
    } catch (const std::out_of_range&) {
        progress_rejected = true;
    }
    assert(progress_rejected);
    return 0;
}
