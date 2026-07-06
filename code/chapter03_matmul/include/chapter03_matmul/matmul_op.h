#pragma once

#include "chapter03_matmul/operator.h"

namespace chapter03_matmul {

class MatMulOp final : public Operator {
public:
    Matrix compute(const Matrix& lhs, const Matrix& rhs) const override;
};

Matrix cpu_reference_matmul(const Matrix& lhs, const Matrix& rhs);

}  // namespace chapter03_matmul
