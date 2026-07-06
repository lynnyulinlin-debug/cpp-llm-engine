#include "chapter03_matmul/matmul_op.h"

#include <stdexcept>

namespace chapter03_matmul {

Matrix cpu_reference_matmul(const Matrix& lhs, const Matrix& rhs) {
    if (!lhs.ready() || !rhs.ready()) {
        throw std::invalid_argument("matrix is not ready");
    }
    if (lhs.cols != rhs.rows) {
        throw std::invalid_argument("shape mismatch");
    }

    Matrix result(lhs.rows, rhs.cols, 0.0f);
    for (std::size_t i = 0; i < lhs.rows; ++i) {
        for (std::size_t k = 0; k < lhs.cols; ++k) {
            const float lhs_value = lhs.at(i, k);
            for (std::size_t j = 0; j < rhs.cols; ++j) {
                result.at(i, j) += lhs_value * rhs.at(k, j);
            }
        }
    }
    return result;
}

Matrix MatMulOp::compute(const Matrix& lhs, const Matrix& rhs) const {
    return cpu_reference_matmul(lhs, rhs);
}

}  // namespace chapter03_matmul
