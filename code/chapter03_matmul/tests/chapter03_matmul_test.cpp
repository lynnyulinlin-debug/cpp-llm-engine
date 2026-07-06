#include "chapter03_matmul/matmul_op.h"

#include <cassert>

void chapter03_matmul_run_operator_smoke_test();

int main() {
    chapter03_matmul_run_operator_smoke_test();

    chapter03_matmul::Matrix lhs(2, 3);
    lhs.at(0, 0) = 1.0f;
    lhs.at(0, 1) = 2.0f;
    lhs.at(0, 2) = 3.0f;
    lhs.at(1, 0) = 4.0f;
    lhs.at(1, 1) = 5.0f;
    lhs.at(1, 2) = 6.0f;

    chapter03_matmul::Matrix rhs(3, 2);
    rhs.at(0, 0) = 7.0f;
    rhs.at(0, 1) = 8.0f;
    rhs.at(1, 0) = 9.0f;
    rhs.at(1, 1) = 10.0f;
    rhs.at(2, 0) = 11.0f;
    rhs.at(2, 1) = 12.0f;

    const chapter03_matmul::Matrix out = chapter03_matmul::cpu_reference_matmul(lhs, rhs);
    assert(out.rows == 2);
    assert(out.cols == 2);
    assert(out.at(0, 0) == 58.0f);
    assert(out.at(0, 1) == 64.0f);
    assert(out.at(1, 0) == 139.0f);
    assert(out.at(1, 1) == 154.0f);
    return 0;
}
