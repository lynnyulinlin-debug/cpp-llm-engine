// Chapter 09：测试 transport-neutral HTTP adapter 的 endpoint 和错误映射。
// 职责：验证 HTTP body 不绕过 ServiceCore，metrics 能观察请求结果。
// 验证入口：Chapter 09 CTest target 或 scripts/run.sh。
#include "chapter05_generation/embedding.h"
#include "chapter05_generation/model_executor.h"
#include "chapter05_generation/position_encoding.h"
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/sampler.h"
#include "chapter05_generation/transformer.h"
#include "chapter09_serving/http_adapter.h"

#include <cassert>
#include <memory>
#include <string>

namespace {

class FixedHttpExecutor final : public chapter05_generation::RequestModelExecutor {
public:
    bool prefill(const chapter05_generation::TokenSequence&,
                 chapter05_generation::Logits* logits,
                 std::string*) override {
        *logits = fixed_logits();
        return true;
    }

    bool decode(chapter05_generation::TokenId,
                chapter05_generation::Logits* logits,
                std::string*) override {
        *logits = fixed_logits();
        return true;
    }

private:
    static chapter05_generation::Logits fixed_logits() {
        chapter05_generation::Logits logits(8, 0.0f);
        logits[3] = 1.0f;
        return logits;
    }
};

}  // namespace

int main() {
    using namespace chapter05_generation;
    using namespace chapter09_serving;

    GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 32;
    Embedding embedding(32);
    PositionEncoding positions(16);
    Transformer transformer(2, embedding, positions);
    Runtime runtime(config, transformer);
    GreedySampler sampler;
    ServingCore service(2, 2);
    HttpAdapter adapter(service, runtime, sampler);

    const HttpResponse health = adapter.handle({"GET", "/health", ""});
    assert(health.status_code == 200 && health.body == "{\"status\":\"ok\"}");

    const HttpResponse invalid = adapter.handle({"POST", "/v1/completions", "{}"});
    assert(invalid.status_code == 400);

    const HttpResponse completion = adapter.handle({
        "POST", "/v1/completions", "{\"request_id\":7,\"prompt_tokens\":[1,2,3]}"});
    assert(completion.status_code == 200);
    assert(completion.body.find("\"finish_reason\":\"stop\"") != std::string::npos);

    const HttpResponse stream = adapter.handle({
        "POST", "/v1/completions", "{\"request_id\":8,\"prompt_tokens\":[1,2],\"stream\":true}"});
    assert(stream.status_code == 200);
    assert(stream.content_type == "text/event-stream");
    assert(stream.body.find("data: {\"token\":") != std::string::npos);
    assert(stream.body.find("finish_reason\":\"stop") != std::string::npos);

    HttpRequest disconnected{
        "POST", "/v1/completions", "{\"request_id\":9,\"prompt_tokens\":[1,2],\"stream\":true}"};
    disconnected.disconnect_after_tokens = 1;
    const HttpResponse disconnect_response = adapter.handle(disconnected);
    assert(disconnect_response.status_code == 499);
    assert(disconnect_response.body.find("client_disconnect") != std::string::npos);

    HttpRequest timed_out{
        "POST", "/v1/completions", "{\"request_id\":10,\"prompt_tokens\":[1,2],\"stream\":true}"};
    timed_out.timeout_after_tokens = 1;
    const HttpResponse timeout_response = adapter.handle(timed_out);
    assert(timeout_response.status_code == 408);
    assert(timeout_response.body.find("finish_reason\":\"timeout") != std::string::npos);

    const HttpResponse metrics = adapter.handle({"GET", "/metrics", ""});
    assert(metrics.status_code == 200);
    assert(metrics.content_type.find("text/plain") != std::string::npos);
    assert(metrics.body.find("serving_accepted_total 4") != std::string::npos);
    assert(metrics.body.find("serving_completed_total 2") != std::string::npos);
    assert(metrics.body.find("serving_failed_total 0") != std::string::npos);
    assert(metrics.body.find("serving_released_total 4") != std::string::npos);
    assert(metrics.body.find("# TYPE serving_queue_wait_seconds summary") != std::string::npos);
    assert(metrics.body.find("serving_queue_wait_seconds_count 4") != std::string::npos);
    assert(metrics.body.find("serving_ttft_seconds_count 4") != std::string::npos);
    assert(metrics.body.find("serving_completion_seconds_count 4") != std::string::npos);

    const HttpResponse missing = adapter.handle({"GET", "/missing", ""});
    assert(missing.status_code == 404);

    Runtime injected_runtime(config);
    ServingCore injected_service(2, 2);
    std::size_t factory_calls = 0;
    HttpAdapter injected_adapter(
        injected_service,
        injected_runtime,
        sampler,
        [&factory_calls]() {
            ++factory_calls;
            return std::make_shared<FixedHttpExecutor>();
        });
    const HttpResponse injected = injected_adapter.handle({
        "POST", "/v1/completions", "{\"request_id\":20,\"prompt_tokens\":[1,2]}"});
    assert(injected.status_code == 200);
    assert(injected.body.find("\"generated_token_ids\":[3,3]") != std::string::npos);
    assert(factory_calls == 1);

    HttpAdapter unavailable_adapter(
        injected_service,
        injected_runtime,
        sampler,
        []() -> std::shared_ptr<RequestModelExecutor> { return {}; });
    const HttpResponse unavailable = unavailable_adapter.handle({
        "POST", "/v1/completions", "{\"request_id\":21,\"prompt_tokens\":[1]}"});
    assert(unavailable.status_code == 500);
    assert(unavailable.body.find("model_executor_unavailable") != std::string::npos);
    return 0;
}
