// Core：无异常状态结果
// 职责：在 backend 和 kernel 之间传递成功标记与错误信息。
// 契约：成功状态没有错误文本，失败状态提供可读 message。
// 验证：core contract 及各章节非法输入测试。
#pragma once

#include <string>
#include <utility>

namespace cpp_llm_engine::core {

class Status final {
public:
    static Status ok() { return Status(true, {}); }
    static Status error(std::string message) { return Status(false, std::move(message)); }

    bool is_ok() const { return ok_; }
    const std::string& message() const { return message_; }

private:
    Status(bool ok, std::string message) : ok_(ok), message_(std::move(message)) {}

    bool ok_ = false;
    std::string message_;
};

}  // namespace cpp_llm_engine::core
