// Core：Operator 公共身份契约
// 职责：为上层算子提供稳定名称，具体输入输出由章节算子定义。
// 契约：实现 name()，backend 选择不放入该基类。
// 验证：code/core/tests/core_contract_test.cpp。
#pragma once

#include "status.h"

#include <string_view>

namespace cpp_llm_engine::core {

class Operator {
public:
    virtual ~Operator() = default;
    virtual std::string_view name() const = 0;
};

}  // namespace cpp_llm_engine::core
