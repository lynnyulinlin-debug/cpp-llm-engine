// Chapter 03：MatMul 数值与边界测试
// 职责：验证矩阵结果、shape mismatch、空请求、越界访问和超大 shape。
// 通过条件：所有 assert 成立，Chapter 03 与 core CTest 均通过。
// 验证：bash code/chapter03_matmul/scripts/run.sh。
#include "chapter03_matmul/matmul_op.h"
#include "chapter03_matmul/cpu_kernel.h"

#include <cassert>
#include <limits>
#include <stdexcept>

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
    const auto out_contract = out.contract();
    assert(out_contract.shape.dimensions == std::vector<std::size_t>({2, 2}));
    assert(out_contract.strides == std::vector<std::size_t>({2, 1}));
    assert(out_contract.validate_element_view(sizeof(float)).is_ok());

    chapter03_matmul::MatMulOp op;
    chapter03_matmul::Matrix explicit_output;
    const auto compute_status = op.try_compute(lhs, rhs, &explicit_output);
    assert(compute_status.is_ok());
    assert(explicit_output.at(1, 1) == 154.0f);
    const auto mismatch_status = op.try_compute(
        chapter03_matmul::Matrix(2, 3), chapter03_matmul::Matrix(2, 2),
        &explicit_output);
    assert(!mismatch_status.is_ok());
    assert(mismatch_status.message() == "matmul shape mismatch");
    assert(!op.try_compute(lhs, rhs, nullptr).is_ok());

    bool threw = false;
    try {
        (void)chapter03_matmul::cpu_reference_matmul(
            chapter03_matmul::Matrix(2, 3), chapter03_matmul::Matrix(2, 2));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    const auto null_status = chapter03_matmul::cpu_reference_matmul_kernel({});
    assert(!null_status.is_ok());

    chapter03_matmul::Matrix one(1, 1, 9.0f);
    bool index_threw = false;
    try {
        (void)one.at(1, 0);
    } catch (const std::out_of_range&) {
        index_threw = true;
    }
    assert(index_threw);

    bool shape_threw = false;
    try {
        chapter03_matmul::Matrix too_large(
            std::numeric_limits<std::size_t>::max(), 2);
    } catch (const std::length_error&) {
        shape_threw = true;
    }
    assert(shape_threw);
    return 0;
}
