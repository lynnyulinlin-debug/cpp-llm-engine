// Chapter 09：验证 HttpAdapter 通过 ResponseWriter 输出真实 token event。
// 职责：覆盖正常 stream、disconnect、timeout 和背压，不依赖具体 socket 或网络库。
// 验证入口：Chapter 09 CTest target 或 scripts/run.sh。
#include "chapter05_generation/embedding.h"
#include "chapter05_generation/position_encoding.h"
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/sampler.h"
#include "chapter05_generation/transformer.h"
#include "chapter09_serving/http_adapter.h"

#include <cassert>
#include <string>

namespace {

class RecordingWriter final : public chapter09_serving::ResponseWriter {
public:
    bool write_headers(const chapter09_serving::ResponseHead& head) override {
        headers = head;
        headers_written = true;
        return connected;
    }

    chapter09_serving::WriteResult write_event(std::string_view event) override {
        if (!connected) return chapter09_serving::WriteResult::ClientClosed;
        if (blocked) return chapter09_serving::WriteResult::Backpressure;
        body += event;
        return chapter09_serving::WriteResult::Ok;
    }

    void finish() override { finished = true; }
    bool client_connected() const override { return connected; }

    chapter09_serving::ResponseHead headers;
    std::string body;
    bool connected = true;
    bool blocked = false;
    bool headers_written = false;
    bool finished = false;
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
    ServingCore service(4, 4);
    HttpAdapter adapter(service, runtime, sampler);

    RecordingWriter normal;
    const HttpResponse normal_response = adapter.handle_stream({
        "POST", "/v1/completions", "{\"request_id\":11,\"prompt_tokens\":[1,2]}"}, normal);
    assert(normal_response.status_code == 200);
    assert(normal.headers_written);
    assert(normal.headers.status_code == 200);
    assert(normal.headers.content_type == "text/event-stream");
    assert(normal.body.find("data: {\"token\":") != std::string::npos);
    assert(normal.body.find("finish_reason\":\"stop") != std::string::npos);
    assert(normal.finished);

    RecordingWriter disconnected;
    HttpRequest disconnect_request{
        "POST", "/v1/completions", "{\"request_id\":12,\"prompt_tokens\":[1,2]}"};
    disconnect_request.disconnect_after_tokens = 1;
    const HttpResponse disconnect_response = adapter.handle_stream(disconnect_request, disconnected);
    assert(disconnect_response.status_code == 499);
    assert(disconnected.body.find("finish_reason\":\"client_disconnect") != std::string::npos);

    RecordingWriter timed_out;
    HttpRequest timeout_request{
        "POST", "/v1/completions", "{\"request_id\":13,\"prompt_tokens\":[1,2]}"};
    timeout_request.timeout_after_tokens = 1;
    const HttpResponse timeout_response = adapter.handle_stream(timeout_request, timed_out);
    assert(timeout_response.status_code == 408);
    assert(timed_out.body.find("finish_reason\":\"timeout") != std::string::npos);

    RecordingWriter backpressured;
    backpressured.blocked = true;
    const HttpResponse backpressure_response = adapter.handle_stream({
        "POST", "/v1/completions", "{\"request_id\":14,\"prompt_tokens\":[1,2]}"}, backpressured);
    assert(backpressure_response.status_code == 503);
    assert(backpressured.body.empty());
    assert(backpressured.finished);
    return 0;
}
