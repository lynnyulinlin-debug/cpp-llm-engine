// Chapter 03：Operator/backend smoke test
// 职责：验证 MatMulOp 名称、CPU backend 身份和最小 1x1 计算。
// 通过条件：公共 Operator 入口返回 2*3=6。
// 验证：chapter03_matmul_test 目标。
#include "chapter03_matmul/matmul_op.h"
#include "chapter03_matmul/cpu_backend.h"

#include <cassert>

namespace {

void run_operator_smoke_test() {
    chapter03_matmul::Matrix lhs(1, 1, 2.0f);
    chapter03_matmul::Matrix rhs(1, 1, 3.0f);
    chapter03_matmul::MatMulOp op;
    assert(op.name() == "matmul");
    assert(op.backend_kind() == cpp_llm_engine::core::BackendKind::CPU);
    const chapter03_matmul::Matrix out = op.compute(lhs, rhs);
    assert(out.rows == 1);
    assert(out.cols == 1);
    assert(out.at(0, 0) == 6.0f);
}

}  // namespace

void chapter03_matmul_run_operator_smoke_test() {
    run_operator_smoke_test();
}
