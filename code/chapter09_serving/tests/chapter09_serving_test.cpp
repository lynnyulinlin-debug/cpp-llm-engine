// Chapter 09：测试 Service Core 的 admission、batch dispatch 和资源回收。
// 职责：验证服务队列复用 Chapter 05 Runtime，同时保持请求状态和 cache 隔离。
// 验证入口：Chapter 09 CTest target 或 scripts/run.sh。
#include "chapter05_generation/embedding.h"
#include "chapter05_generation/model_executor.h"
#include "chapter05_generation/position_encoding.h"
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/sampler.h"
#include "chapter05_generation/serving_preview.h"
#include "chapter05_generation/transformer.h"
#include "chapter09_serving/serving_core.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace {

class ServiceExecutor final : public chapter05_generation::RequestModelExecutor {
public:
    explicit ServiceExecutor(chapter05_generation::TokenId token) : token_(token) {}

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

class FailingServiceExecutor final : public chapter05_generation::RequestModelExecutor {
public:
    bool prefill(const chapter05_generation::TokenSequence&,
                 chapter05_generation::Logits*,
                 std::string* error) override {
        *error = "service model failure";
        return false;
    }

    bool decode(chapter05_generation::TokenId,
                chapter05_generation::Logits*,
                std::string*) override {
        return false;
    }
};

class ChunkedServiceExecutor final : public chapter05_generation::RequestModelExecutor {
public:
    bool prefill(const chapter05_generation::TokenSequence& prompt,
                 chapter05_generation::Logits* logits,
                 std::string* error) override {
        return prefill_chunk(prompt, true, true, logits, error);
    }

    bool supports_chunked_prefill() const override { return true; }

    bool prefill_chunk(const chapter05_generation::TokenSequence&,
                       bool,
                       bool produce_logits,
                       chapter05_generation::Logits* logits,
                       std::string*) override {
        if (produce_logits) {
            logits->assign(8, 0.0f);
            (*logits)[3] = 1.0f;
        }
        return true;
    }

    bool decode(chapter05_generation::TokenId,
                chapter05_generation::Logits* logits,
                std::string*) override {
        logits->assign(8, 0.0f);
        (*logits)[3] = 1.0f;
        return true;
    }
};

class ManualClock final {
public:
    chapter09_serving::ServiceClock::time_point now() const { return now_; }

    void advance(std::chrono::milliseconds duration) { now_ += duration; }

private:
    chapter09_serving::ServiceClock::time_point now_{};
};

bool near(double left, double right) {
    return std::abs(left - right) < 1e-12;
}

}  // namespace

int main() {
    using namespace chapter05_generation;
    using namespace chapter09_serving;

    bool invalid_config = false;
    try {
        ServingCore invalid(1, 2);
    } catch (const std::invalid_argument&) {
        invalid_config = true;
    }
    assert(invalid_config);

    GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 32;
    Embedding embedding(32);
    PositionEncoding positions(16);
    Transformer transformer(2, embedding, positions);
    Runtime runtime(config, transformer);
    GreedySampler sampler;
    Request request_a({1, 2});
    Request request_b({3, 4});
    Request request_c({5, 6});
    Request cancelled({7});
    ServingCore service(3, 2);

    assert(service.admit(1, request_a, sampler).accepted());
    assert(service.admit(2, request_b, sampler).accepted());
    assert(service.admit(3, request_c, sampler).accepted());
    assert(service.admit(1, request_c, sampler).status == AdmissionStatus::DuplicateId);
    Request rejected({8});
    assert(service.admit(4, rejected, sampler).status == AdmissionStatus::QueueFull);
    assert(service.waiting_size() == 3);
    assert(service.cancel(3));
    assert(request_c.cancelled());
    assert(service.waiting_size() == 2);
    assert(service.metrics().accepted_total == 3);
    assert(service.metrics().duplicate_total == 1);
    assert(service.metrics().queue_full_total == 1);
    assert(service.metrics().cancelled_total == 1);
    assert(service.metrics().peak_inflight == 3);

    const ServiceStats first = service.dispatch(runtime);
    assert(first.admitted == 2);
    assert(first.completed == 2);
    assert(first.generated_tokens == 4);
    assert(request_a.finished() && request_b.finished());
    assert(service.running_size() == 2);
    assert(service.release_completed() == 2);
    assert(request_a.cache().empty() && request_b.cache().empty());
    assert(service.running_size() == 0);
    assert(service.metrics().completed_total == 2);
    assert(service.metrics().released_total == 2);
    assert(service.metrics().generated_tokens_total == 4);

    assert(service.admit(5, cancelled, sampler).accepted());
    assert(service.timeout(5));
    assert(cancelled.timed_out());
    assert(service.metrics().timed_out_total == 1);
    assert(service.waiting_size() == 0);
    assert(service.dispatch(runtime).completed == 0);
    assert(service.metrics().queue_wait_count == 2);
    assert(service.metrics().ttft_count == 2);
    assert(service.metrics().completion_count == 4);

    Runtime executor_runtime(config);
    ServingCore executor_service(2, 2);
    Request executor_request_a({1});
    Request executor_request_b({2});
    auto executor_a = std::make_shared<ServiceExecutor>(3);
    auto executor_b = std::make_shared<ServiceExecutor>(4);
    assert(executor_service.admit(10, executor_request_a, sampler, executor_a).accepted());
    assert(executor_service.admit(11, executor_request_b, sampler, executor_b).accepted());
    const auto executor_stats = executor_service.dispatch(executor_runtime);
    assert(executor_stats.completed == 2);
    assert(executor_stats.generated_tokens == 4);
    assert(executor_request_a.generated() == TokenSequence({3, 3}));
    assert(executor_request_b.generated() == TokenSequence({4, 4}));
    assert(executor_a->prefill_calls == 1 && executor_a->decode_calls == 1);
    assert(executor_b->prefill_calls == 1 && executor_b->decode_calls == 1);
    assert(executor_service.release_completed() == 2);
    Request failed_request({5});
    auto failing_executor = std::make_shared<FailingServiceExecutor>();
    assert(executor_service.admit(12, failed_request, sampler, failing_executor).accepted());
    assert(executor_service.dispatch(executor_runtime).generated_tokens == 0);
    assert(failed_request.failed());
    assert(failed_request.failure_reason() == "service model failure");
    assert(executor_service.release_completed() == 1);
    assert(executor_service.metrics().failed_total == 1);
    assert(executor_service.metrics().queue_wait_count == 3);
    assert(executor_service.metrics().ttft_count == 2);
    assert(executor_service.metrics().completion_count == 3);

    auto step_pool = std::make_shared<KvBlockPool>(2, 2);
    ServingCore step_service(3, 2, ServiceNow{}, step_pool);
    Request step_request_a({1});
    Request step_request_b({2});
    Request step_request_c({3});
    auto step_executor_a = std::make_shared<ServiceExecutor>(3);
    auto step_executor_b = std::make_shared<ServiceExecutor>(4);
    auto step_executor_c = std::make_shared<ServiceExecutor>(5);
    assert(step_service.admit(
        30, step_request_a, sampler, step_executor_a).accepted());
    assert(step_service.admit(
        31, step_request_b, sampler, step_executor_b).accepted());
    bool blocked_legacy_dispatch = false;
    try {
        (void)step_service.dispatch(executor_runtime);
    } catch (const std::logic_error&) {
        blocked_legacy_dispatch = true;
    }
    assert(blocked_legacy_dispatch);
    const auto first_step = step_service.dispatch_step(executor_runtime, 2);
    assert(first_step.admitted == 2);
    assert(first_step.generated_tokens == 2);
    assert(first_step.completed == 0);
    assert(first_step.scheduled_requests == 2);
    assert(first_step.scheduled_tokens == 2);
    assert(first_step.deferred_requests == 0);
    assert(step_pool->free_blocks() == 0);
    assert(step_service.admit(
        32, step_request_c, sampler, step_executor_c).accepted());
    assert(step_service.waiting_size() == 1);
    const auto second_step = step_service.dispatch_step(executor_runtime, 2);
    assert(second_step.generated_tokens == 2);
    assert(second_step.completed == 2);
    assert(step_service.waiting_size() == 1);
    assert(step_service.release_completed() == 2);
    assert(step_pool->free_blocks() == 2);
    const auto replacement_step = step_service.dispatch_step(executor_runtime, 1);
    assert(replacement_step.admitted == 1);
    assert(replacement_step.generated_tokens == 1);
    assert(step_service.waiting_size() == 0);
    assert(!step_request_c.terminal());
    assert(step_service.dispatch_step(executor_runtime, 1).completed == 1);
    assert(step_service.release_completed() == 1);
    assert(step_pool->free_blocks() == 2);
    assert(step_service.metrics().generated_tokens_total == 6);
    assert(step_service.metrics().completed_total == 3);

    auto cancelled_pool = std::make_shared<KvBlockPool>(3, 2);
    ServingCore cancelled_service(1, 1, ServiceNow{}, cancelled_pool);
    Request partial_request({1, 2, 3, 4, 5});
    auto partial_executor = std::make_shared<ChunkedServiceExecutor>();
    assert(cancelled_service.admit(
        40, partial_request, sampler, partial_executor).accepted());
    const auto partial_step = cancelled_service.dispatch_step(executor_runtime, 2);
    assert(partial_step.generated_tokens == 0 && partial_step.scheduled_tokens == 2);
    assert(partial_request.prefill_offset() == 2);
    assert(cancelled_pool->token_count(40) == 2);
    assert(cancelled_service.cancel(40));
    assert(cancelled_service.release_completed() == 1);
    assert(partial_request.cancelled() && partial_request.cache().empty());
    assert(cancelled_pool->free_blocks() == 3 && !cancelled_pool->contains(40));

    ManualClock clock;
    ServingCore latency_service(2, 1, [&clock]() { return clock.now(); });
    Request latency_request_a({1});
    Request latency_request_b({2});
    auto latency_executor_a = std::make_shared<ServiceExecutor>(3);
    auto latency_executor_b = std::make_shared<ServiceExecutor>(4);
    assert(latency_service.admit(
        20, latency_request_a, sampler, latency_executor_a).accepted());
    clock.advance(std::chrono::milliseconds(4));
    assert(latency_service.admit(
        21, latency_request_b, sampler, latency_executor_b).accepted());
    clock.advance(std::chrono::milliseconds(6));
    const auto advance_after_token = [&clock](std::size_t, TokenId, std::size_t) {
        clock.advance(std::chrono::milliseconds(3));
    };
    assert(latency_service.dispatch(executor_runtime, advance_after_token).completed == 1);
    assert(latency_service.release_completed() == 1);
    assert(latency_service.dispatch(executor_runtime, advance_after_token).completed == 1);
    assert(latency_service.release_completed() == 1);
    const auto& latency = latency_service.metrics();
    assert(latency.queue_wait_count == 2);
    assert(latency.ttft_count == 2);
    assert(latency.completion_count == 2);
    assert(near(latency.queue_wait_seconds_sum, 0.022));
    assert(near(latency.ttft_seconds_sum, 0.022));
    assert(near(latency.completion_seconds_sum, 0.028));
    return 0;
}
