// Engine 集成测试：验证预计算 embedding 复用 Chapter09 同步与 SSE 服务链路。
// 覆盖：每请求 Executor、token 一致性、失败终态、metrics 和 Context 释放。
// 依赖：使用合成 Decoder Context，不依赖模型、图片、socket、Python 或 GPU。
// 运行：由 Engine CTest 执行 embedding_request_service_test。
#include "cpp_llm_engine/embedding_request_executor.h"

#include "chapter05_generation/runtime.h"
#include "chapter05_generation/sampler.h"
#include "chapter09_serving/http_adapter.h"

#include <cassert>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct ContextState {
    std::size_t embedding_prefill_calls = 0;
    std::size_t decode_calls = 0;
};

class ServiceEmbeddingContext final
    : public cpp_llm_engine::core::DecoderExecutionContext {
public:
    ServiceEmbeddingContext(std::shared_ptr<ContextState> state, bool fail_prefill)
        : state_(std::move(state)), fail_prefill_(fail_prefill) {}

    void reset() override { token_count_ = 0; }

    cpp_llm_engine::core::Status prefill_tokens(
        const cpp_llm_engine::core::DecoderTokenSequence&,
        cpp_llm_engine::core::DecoderLogits*) override {
        return cpp_llm_engine::core::Status::error("token prefill is not expected");
    }

    cpp_llm_engine::core::Status prefill_embeddings(
        const cpp_llm_engine::core::DecoderEmbeddingSequence& embeddings,
        cpp_llm_engine::core::DecoderLogits* logits) override {
        ++state_->embedding_prefill_calls;
        if (fail_prefill_) {
            return cpp_llm_engine::core::Status::error("synthetic visual prefill failure");
        }
        token_count_ = embeddings.token_count;
        logits->assign(5, 0.0f);
        (*logits)[2] = 1.0f;
        return cpp_llm_engine::core::Status::ok();
    }

    cpp_llm_engine::core::Status decode_token(
        cpp_llm_engine::core::DecoderTokenId,
        cpp_llm_engine::core::DecoderLogits* logits) override {
        ++state_->decode_calls;
        ++token_count_;
        logits->assign(5, 0.0f);
        (*logits)[4] = 1.0f;
        return cpp_llm_engine::core::Status::ok();
    }

    std::size_t token_count() const override { return token_count_; }

    cpp_llm_engine::core::DecoderModelContract model_contract() const override {
        cpp_llm_engine::core::DecoderLayerContract layer;
        layer.hidden_size = 4;
        layer.ffn_hidden_size = 8;
        layer.query_head_count = 1;
        layer.kv_head_count = 1;
        layer.head_dimension = 4;
        return {5, 2, 16, layer};
    }

private:
    std::shared_ptr<ContextState> state_;
    bool fail_prefill_ = false;
    std::size_t token_count_ = 0;
};

class RecordingWriter final : public chapter09_serving::ResponseWriter {
public:
    bool write_headers(const chapter09_serving::ResponseHead& value) override {
        head = value;
        return true;
    }

    chapter09_serving::WriteResult write_event(std::string_view event) override {
        body += event;
        return chapter09_serving::WriteResult::Ok;
    }

    void finish() override { finished = true; }
    bool client_connected() const override { return true; }

    chapter09_serving::ResponseHead head;
    std::string body;
    bool finished = false;
};

cpp_llm_engine::core::DecoderEmbeddingSequence make_embeddings() {
    return {3, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                   5.0f, 6.0f, 7.0f, 8.0f,
                   9.0f, 10.0f, 11.0f, 12.0f}};
}

std::shared_ptr<chapter05_generation::RequestModelExecutor> make_executor(
    bool fail_prefill,
    std::vector<std::weak_ptr<ContextState>>* observed_states) {
    auto state = std::make_shared<ContextState>();
    observed_states->push_back(state);
    auto context = std::make_unique<ServiceEmbeddingContext>(state, fail_prefill);
    std::string error;
    auto executor = cpp_llm_engine::EmbeddingRequestExecutor::create(
        std::move(context), make_embeddings(), &error);
    assert(executor != nullptr && error.empty());
    return executor;
}

}  // namespace

int main() {
    chapter05_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 16;
    config.eos_token_id = 4;
    chapter05_generation::Runtime runtime(config);
    chapter05_generation::GreedySampler sampler;
    chapter09_serving::ServingCore service(2, 2);
    std::vector<std::weak_ptr<ContextState>> observed_states;
    chapter09_serving::HttpAdapter adapter(
        service,
        runtime,
        sampler,
        [&observed_states]() { return make_executor(false, &observed_states); });

    const auto sync = adapter.handle({
        "POST", "/v1/completions",
        "{\"request_id\":71,\"prompt_tokens\":[0,0,0]}"});
    assert(sync.status_code == 200);
    assert(sync.body.find("\"generated_token_ids\":[2,4]") != std::string::npos);
    assert(observed_states.size() == 1 && observed_states[0].expired());

    RecordingWriter writer;
    const auto stream = adapter.handle_stream({
        "POST", "/v1/completions",
        "{\"request_id\":72,\"prompt_tokens\":[0,0,0]}"}, writer);
    assert(stream.status_code == 200);
    assert(writer.head.status_code == 200);
    assert(writer.head.content_type == "text/event-stream");
    assert(writer.body.find("data: {\"token\":2,\"step\":0}") != std::string::npos);
    assert(writer.body.find("data: {\"token\":4,\"step\":1}") != std::string::npos);
    assert(writer.body.find("finish_reason\":\"stop") != std::string::npos);
    assert(writer.finished);
    assert(observed_states.size() == 2 && observed_states[1].expired());
    assert(service.metrics().accepted_total == 2);
    assert(service.metrics().completed_total == 2);
    assert(service.metrics().released_total == 2);
    assert(service.metrics().generated_tokens_total == 4);

    chapter09_serving::ServingCore failing_service(1, 1);
    std::vector<std::weak_ptr<ContextState>> failed_states;
    chapter09_serving::HttpAdapter failing_adapter(
        failing_service,
        runtime,
        sampler,
        [&failed_states]() { return make_executor(true, &failed_states); });
    const auto failed = failing_adapter.handle({
        "POST", "/v1/completions",
        "{\"request_id\":73,\"prompt_tokens\":[0,0,0]}"});
    assert(failed.status_code == 500);
    assert(failed.body.find("runtime_did_not_finish") != std::string::npos);
    assert(failing_service.metrics().failed_total == 1);
    assert(failing_service.metrics().released_total == 1);
    assert(failed_states.size() == 1 && failed_states[0].expired());

    chapter09_serving::ServingCore stress_service(4, 4);
    std::vector<std::weak_ptr<ContextState>> stress_states;
    chapter09_serving::HttpAdapter stress_adapter(
        stress_service,
        runtime,
        sampler,
        [&stress_states]() { return make_executor(false, &stress_states); });
    constexpr std::size_t kStressRequests = 32;
    for (std::size_t index = 0; index < kStressRequests; ++index) {
        const std::string body = "{\"request_id\":" + std::to_string(1000 + index) +
            ",\"prompt_tokens\":[0,0,0]}";
        if (index % 2 == 0) {
            const auto response = stress_adapter.handle(
                {"POST", "/v1/completions", body});
            assert(response.status_code == 200);
            assert(response.body.find("\"generated_token_ids\":[2,4]") !=
                   std::string::npos);
        } else {
            RecordingWriter stress_writer;
            const auto response = stress_adapter.handle_stream(
                {"POST", "/v1/completions", body}, stress_writer);
            assert(response.status_code == 200);
            assert(stress_writer.body.find("data: {\"token\":2,\"step\":0}") !=
                   std::string::npos);
            assert(stress_writer.body.find("data: {\"token\":4,\"step\":1}") !=
                   std::string::npos);
            assert(stress_writer.finished);
        }
        assert(stress_states.size() == index + 1);
        assert(stress_states.back().expired());
        assert(stress_service.inflight_size() == 0);
    }
    assert(stress_service.metrics().accepted_total == kStressRequests);
    assert(stress_service.metrics().completed_total == kStressRequests);
    assert(stress_service.metrics().released_total == kStressRequests);
    assert(stress_service.metrics().generated_tokens_total == kStressRequests * 2);

    chapter09_serving::ServingCore stress_failure_service(2, 2);
    std::vector<std::weak_ptr<ContextState>> stress_failure_states;
    chapter09_serving::HttpAdapter stress_failure_adapter(
        stress_failure_service,
        runtime,
        sampler,
        [&stress_failure_states]() {
            return make_executor(true, &stress_failure_states);
        });
    constexpr std::size_t kFailureRequests = 8;
    for (std::size_t index = 0; index < kFailureRequests; ++index) {
        const std::string body = "{\"request_id\":" + std::to_string(2000 + index) +
            ",\"prompt_tokens\":[0,0,0]}";
        const auto response = stress_failure_adapter.handle(
            {"POST", "/v1/completions", body});
        assert(response.status_code == 500);
        assert(stress_failure_states.back().expired());
        assert(stress_failure_service.inflight_size() == 0);
    }
    assert(stress_failure_service.metrics().accepted_total == kFailureRequests);
    assert(stress_failure_service.metrics().failed_total == kFailureRequests);
    assert(stress_failure_service.metrics().released_total == kFailureRequests);

    chapter09_serving::ServingCore terminal_service(2, 2);
    std::vector<std::weak_ptr<ContextState>> terminal_states;
    chapter09_serving::HttpAdapter terminal_adapter(
        terminal_service,
        runtime,
        sampler,
        [&terminal_states]() { return make_executor(false, &terminal_states); });
    chapter09_serving::HttpRequest disconnected{
        "POST", "/v1/completions",
        "{\"request_id\":3001,\"prompt_tokens\":[0,0,0]}"};
    disconnected.disconnect_after_tokens = 1;
    const auto disconnected_response = terminal_adapter.handle(disconnected);
    assert(disconnected_response.status_code == 499);
    assert(terminal_states.back().expired());

    chapter09_serving::HttpRequest timed_out{
        "POST", "/v1/completions",
        "{\"request_id\":3002,\"prompt_tokens\":[0,0,0]}"};
    timed_out.timeout_after_tokens = 1;
    const auto timeout_response = terminal_adapter.handle(timed_out);
    assert(timeout_response.status_code == 408);
    assert(terminal_states.back().expired());
    assert(terminal_service.metrics().cancelled_total == 1);
    assert(terminal_service.metrics().timed_out_total == 1);
    assert(terminal_service.metrics().released_total == 2);
    assert(terminal_service.inflight_size() == 0);
    return 0;
}
