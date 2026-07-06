#pragma once

#include <cstddef>
#include <vector>

namespace chapter04_attention {

struct Tensor {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> data;

    Tensor() = default;
    Tensor(std::size_t r, std::size_t c, float value = 0.0f);

    float& at(std::size_t row, std::size_t col);
    float at(std::size_t row, std::size_t col) const;
    bool ready() const;
};

Tensor add(const Tensor& lhs, const Tensor& rhs);
Tensor transpose(const Tensor& tensor);
Tensor matmul(const Tensor& lhs, const Tensor& rhs);
Tensor slice_columns(const Tensor& tensor, std::size_t start_col, std::size_t width);
Tensor concatenate_columns(const std::vector<Tensor>& parts);
void scale_in_place(Tensor& tensor, float factor);

}  // namespace chapter04_attention
