#pragma once

#include "chapter03_matmul/matrix.h"

namespace chapter03_matmul {

class Operator {
public:
    virtual ~Operator() = default;
    virtual Matrix compute(const Matrix& lhs, const Matrix& rhs) const = 0;
};

}  // namespace chapter03_matmul
