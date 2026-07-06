#include "chapter03_matmul/matrix.h"

#include <stdexcept>

namespace chapter03_matmul {

Matrix::Matrix(std::size_t r, std::size_t c, float value)
    : rows(r), cols(c), data(r * c, value) {}

float& Matrix::at(std::size_t row, std::size_t col) {
    return data.at(row * cols + col);
}

float Matrix::at(std::size_t row, std::size_t col) const {
    return data.at(row * cols + col);
}

bool Matrix::ready() const {
    return rows > 0 && cols > 0 && data.size() == rows * cols;
}

}  // namespace chapter03_matmul
