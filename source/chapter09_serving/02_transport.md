# 09.2 Transport 与 C++ 网络库

HTTP/SSE 通过 Transport Adapter 接入 Chapter09。网络库处理连接、请求解析、响应写出和连接事件，`ServiceCore` 处理请求状态、metrics 和 streaming contract。

## 传输层边界

当前 `HttpAdapter::handle()` 继续返回完整响应对象，兼容同步测试；`handle_stream()` 通过 `ResponseWriter` 写出 token event。可选 cpp-httplib transport 已完成真实 loopback socket 验证，默认构建仍只运行 transport-neutral 测试。网络库只实现 writer 和请求解析，不把网络库类型传入 ServiceCore。

```text
cpp-httplib / Boost.Beast
            |
      Transport Adapter
            |
   ResponseWriter / StreamSink
            |
       HttpAdapter
            |
       ServiceCore
            |
 Runtime + metrics + Request KV Cache
```

`ResponseWriter` 只报告写出结果和连接状态，不拥有请求调度状态：

| 能力 | 传输层负责 | ServiceCore 负责 |
| --- | --- | --- |
| 请求解析 | HTTP method、path、body | 参数语义和 admission |
| 响应写出 | headers、SSE event、finish | token event 和终态 |
| 客户端断开 | 报告连接已关闭 | 将 Request 标记为 cancelled 并释放资源 |
| 超时 | 提供 deadline 事件 | 终止请求、停止调度和回收 cache |
| 背压 | 报告无法继续写出 | 决定暂停、取消或记录失败 |
| 指标 | 可选 HTTP 连接指标 | 现有 ServiceCore metrics |

## 网络库比较

| 方案 | 适合当前目标 | 主要优点 | 主要限制 |
| --- | --- | --- | --- |
| `cpp-httplib` | 第一阶段真实 HTTP 闭环 | 单头文件、接入快，支持 HTTP/HTTPS、chunked response 和 SSE | 以阻塞式 HTTP/1.1 为主，不适合作为最终高并发 transport |
| `Boost.Asio + Boost.Beast` | 进阶和生产方向 | 异步 IO、连接生命周期、超时、背压和 TLS 组合能力完整 | 类型和状态机较多，教学实现成本高 |
| Crow | 快速搭建 Web API | 路由和 WebSocket 接口较方便 | 框架抽象较多，容易隐藏 transport 边界 |
| oat++ | 完整 Web 服务 | 异步、多线程、流式响应和 TLS 组件较完整 | 引入 DI、对象映射和框架运行时，超出本章最小目标 |

本教程采用两级方案：先用 `cpp-httplib` 验证真实 loopback HTTP/SSE，再用 `Boost.Asio + Boost.Beast` 验证异步 transport。两种实现都必须复用同一个 `HttpAdapter`、`ServiceCore`、metrics 和 streaming contract。

## `ResponseWriter` 契约

真实网络适配器不应让 `HttpResponse` 保存无限增长的 stream body。当前已经增加独立的写出接口，测试使用 FakeWriter，网络库使用具体 writer：

```cpp
class ResponseWriter {
public:
    virtual bool write_headers(const ResponseHead& head) = 0;
    virtual WriteResult write_event(std::string_view event) = 0;
    virtual void finish() = 0;
    virtual bool client_connected() const = 0;
    virtual ~ResponseWriter() = default;
};
```

`WriteResult` 至少区分 `ok`、`client_closed` 和 `backpressure`。客户端断开仍映射为现有 `499/client_disconnect`，deadline 仍映射为 `408/timeout`；网络库不能自行定义另一套终态。

真实 HTTP 在 headers 写出后不能再修改 status code。因此 `handle_stream()` 返回的 `499` 或 `408` 是适配器记录的语义状态；已经发送 `200` headers 的连接通过最终 event 表达 `client_disconnect` 或 `timeout`，ServiceCore 的 Request 终态和 metrics 保持不变。

## 接入顺序

1. 已增加 `ResponseWriter`/`StreamSink` 和 FakeWriter 测试。
2. 已验证正常输出、SSE 顺序、客户端断开、timeout、背压和 KV Cache 回收。
3. 已加入可选 `cpp-httplib` transport 构建目标，并完成 `/health`、`/metrics`、同步 completion 和 SSE loopback 测试。
4. 已在 Engine 应用组合层把真实 TinyLlama Factory 注入同一个 `HttpAdapter`，不修改 transport。
5. 将网络 IO 与 Runtime 执行线程分开，避免阻塞网络线程。
6. 再增加 Beast/Asio transport，复用同一组 contract tests。
7. 对两种 transport 比较并发、TTFT、吞吐、断开和超时行为。

网络库替换最外层 transport，并通过稳定适配接口复用队列、Request 终态、metrics 字段和 token event 格式。接入差异优先在适配层处理。

## 验证入口

当前 transport-neutral 验证：

```bash
bash code/chapter09_serving/scripts/run.sh
```

当前脚本会运行 `HttpAdapter::handle_stream()` 的 writer 集成测试；`run_cpp_httplib_loopback.sh` 额外验证真实 loopback socket。

真实 loopback 验证：

```bash
CHAPTER09_CPPHTTPLIB_INCLUDE_DIR=/path/to/cpp-httplib \
bash code/chapter09_serving/scripts/run_cpp_httplib_loopback.sh
```

脚本会启动本地 server，依次验证 `/health`、`/metrics`、同步 completion 和 SSE completion。未设置 include path 时，CMake 会尝试获取固定版本 `v0.18.3`。

接入真实网络库后的最低验证集：

| 场景 | 必须观察的结果 |
| --- | --- |
| `/health` | HTTP 200 和健康状态 |
| `/metrics` | ServiceCore 累计指标未改变 |
| 同步 completion | status、body 和 finish reason 正确 |
| SSE completion | token 顺序和最终 finish event 正确 |
| 客户端断开 | `499/client_disconnect`、Request 终止、KV Cache 释放 |
| deadline | `408/timeout`、停止后续 decode、KV Cache 释放 |
| 背压 | 不产生重复 token，不绕过 ServiceCore 终态 |
