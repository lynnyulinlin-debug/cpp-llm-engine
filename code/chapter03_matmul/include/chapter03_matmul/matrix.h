#pragma once

#include <cstddef>
#include <vector>

namespace chapter03_matmul {

struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> data;

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, float value = 0.0f);

    float& at(std::size_t row, std::size_t col);
    float at(std::size_t row, std::size_t col) const;
    bool ready() const;
};

}  // namespace chapter03_matmul
