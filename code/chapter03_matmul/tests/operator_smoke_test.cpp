#include "chapter03_matmul/matmul_op.h"

#include <cassert>

namespace {

void run_operator_smoke_test() {
    chapter03_matmul::Matrix lhs(1, 1, 2.0f);
    chapter03_matmul::Matrix rhs(1, 1, 3.0f);
    chapter03_matmul::MatMulOp op;
    const chapter03_matmul::Matrix out = op.compute(lhs, rhs);
    assert(out.rows == 1);
    assert(out.cols == 1);
    assert(out.at(0, 0) == 6.0f);
}

}  // namespace

void chapter03_matmul_run_operator_smoke_test() {
    run_operator_smoke_test();
}
