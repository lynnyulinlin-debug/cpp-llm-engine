// Chapter 09：声明 cpp-httplib 到 transport-neutral HttpAdapter 的适配器。
// 职责：把真实 HTTP 请求交给 HttpAdapter，把同步响应和 SSE event 写回 socket。
// 验证：启用 CHAPTER09_ENABLE_CPPHTTPLIB 后运行 server，并用 loopback client 验证 endpoint。
#pragma once

#include "chapter09_serving/http_adapter.h"

#include <string>

namespace chapter09_serving {

class CppHttplibServer final {
public:
    CppHttplibServer(HttpAdapter& adapter, std::string bind_address, int port);

    int listen();

private:
    HttpAdapter& adapter_;
    std::string bind_address_;
    int port_;
};

}  // namespace chapter09_serving
