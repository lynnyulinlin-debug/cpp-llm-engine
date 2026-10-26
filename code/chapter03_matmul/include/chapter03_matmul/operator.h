// Chapter 03：章节 Operator 接口
// 职责：为 MatMul 等教学算子提供统一 Matrix compute 入口。
// 契约：上层只依赖输入输出语义，backend 选择由具体算子保存。
// 验证：operator_smoke_test.cpp。
#pragma once

#include "chapter03_matmul/matrix.h"
#include "cpp_llm_engine/core/operator.h"

namespace chapter03_matmul {

class Operator : public cpp_llm_engine::core::Operator {
public:
    virtual ~Operator() = default;
    virtual Matrix compute(const Matrix& lhs, const Matrix& rhs) const = 0;
};

}  // namespace chapter03_matmul
