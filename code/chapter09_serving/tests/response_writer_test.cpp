// Chapter 09：验证 transport-neutral ResponseWriter 的最小生命周期。
// 职责：模拟网络写出、客户端断开和背压，不启动 socket，也不修改 ServiceCore 状态。
// 验证入口：Chapter 09 CTest target 或 scripts/run.sh。
#include "chapter09_serving/response_writer.h"

#include <cassert>
#include <string>

namespace {

class FakeWriter final : public chapter09_serving::ResponseWriter {
public:
    bool write_headers(const chapter09_serving::ResponseHead& head) override {
        if (!connected_ || finished_) return false;
        head_ = head;
        headers_written_ = true;
        return true;
    }

    chapter09_serving::WriteResult write_event(std::string_view event) override {
        if (!connected_) return chapter09_serving::WriteResult::ClientClosed;
        if (blocked_) return chapter09_serving::WriteResult::Backpressure;
        if (finished_ || !headers_written_) return chapter09_serving::WriteResult::ClientClosed;
        body_ += event;
        return chapter09_serving::WriteResult::Ok;
    }

    void finish() override { finished_ = true; }

    bool client_connected() const override { return connected_; }

    void disconnect() { connected_ = false; }
    void set_blocked(bool blocked) { blocked_ = blocked; }

    const chapter09_serving::ResponseHead& head() const { return head_; }
    const std::string& body() const { return body_; }
    bool finished() const { return finished_; }

private:
    chapter09_serving::ResponseHead head_;
    std::string body_;
    bool connected_ = true;
    bool blocked_ = false;
    bool headers_written_ = false;
    bool finished_ = false;
};

}  // namespace

int main() {
    using namespace chapter09_serving;

    FakeWriter writer;
    assert(writer.write_headers({200, "text/event-stream"}));
    assert(writer.head().status_code == 200);
    assert(writer.write_event("token-1\n\n") == WriteResult::Ok);
    assert(writer.body() == "token-1\n\n");

    writer.set_blocked(true);
    assert(writer.write_event("token-2\n\n") == WriteResult::Backpressure);
    assert(writer.body() == "token-1\n\n");

    writer.set_blocked(false);
    writer.disconnect();
    assert(!writer.client_connected());
    assert(writer.write_event("token-3\n\n") == WriteResult::ClientClosed);

    writer.finish();
    assert(writer.finished());
    assert(!writer.write_headers({200, "text/event-stream"}));
    return 0;
}
