// Core：计算 backend 基类
// 职责：提供 CPU/CUDA backend 的设备类型和可观察名称。
// 契约：具体 backend 实现 kind() 和 name()，不在此处执行算子计算。
// 验证：code/core/tests/core_contract_test.cpp。
#pragma once

#include "status.h"

#include <string_view>

namespace cpp_llm_engine::core {

enum class BackendKind {
    CPU,
    CUDA,
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual BackendKind kind() const = 0;
    virtual std::string_view name() const = 0;
};

}  // namespace cpp_llm_engine::core
