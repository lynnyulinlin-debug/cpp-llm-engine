// Chapter 09：声明不绑定具体网络库的流式响应写出契约。
// 职责：统一 headers、token event、连接状态和背压结果，隔离 HttpAdapter 与 socket 实现。
// 验证：response_writer_test 覆盖正常写出、客户端断开、背压和 finish 生命周期。
#pragma once

#include <string>
#include <string_view>

namespace chapter09_serving {

struct ResponseHead {
    int status_code = 500;
    std::string content_type = "application/json";
};

enum class WriteResult {
    Ok,
    ClientClosed,
    Backpressure,
};

class ResponseWriter {
public:
    virtual ~ResponseWriter() = default;

    virtual bool write_headers(const ResponseHead& head) = 0;
    virtual WriteResult write_event(std::string_view event) = 0;
    virtual void finish() = 0;
    virtual bool client_connected() const = 0;
};

}  // namespace chapter09_serving
