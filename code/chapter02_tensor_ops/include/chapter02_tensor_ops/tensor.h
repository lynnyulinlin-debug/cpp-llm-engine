#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace chapter02_tensor_ops {

class Tensor final {
public:
    Tensor();
    Tensor(std::size_t rows, std::size_t cols, float value = 0.0f);

    float& at(std::size_t row, std::size_t col);
    float at(std::size_t row, std::size_t col) const;

    bool ready() const;
    bool contiguous() const;
    bool shares_storage_with(const Tensor& other) const;
    std::size_t storage_use_count() const;

    Tensor reshape(std::size_t rows, std::size_t cols) const;
    Tensor slice(std::size_t row_start, std::size_t row_count, std::size_t col_start, std::size_t col_count) const;

    std::size_t rows() const;
    std::size_t cols() const;

private:
    Tensor(std::shared_ptr<std::vector<float>> storage,
           std::size_t rows,
           std::size_t cols,
           std::size_t offset,
           std::size_t row_stride,
           std::size_t col_stride);

    std::shared_ptr<std::vector<float>> storage_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t offset_ = 0;
    std::size_t row_stride_ = 0;
    std::size_t col_stride_ = 1;
};

}  // namespace chapter02_tensor_ops
