#include "chapter02_tensor_ops/tensor.h"

#include <stdexcept>

namespace chapter02_tensor_ops {

Tensor::Tensor() = default;

Tensor::Tensor(std::size_t rows, std::size_t cols, float value)
    : storage_(std::make_shared<std::vector<float>>(rows * cols, value)),
      rows_(rows),
      cols_(cols),
      offset_(0),
      row_stride_(cols),
      col_stride_(1) {}

Tensor::Tensor(std::shared_ptr<std::vector<float>> storage,
               std::size_t rows,
               std::size_t cols,
               std::size_t offset,
               std::size_t row_stride,
               std::size_t col_stride)
    : storage_(std::move(storage)),
      rows_(rows),
      cols_(cols),
      offset_(offset),
      row_stride_(row_stride),
      col_stride_(col_stride) {}

float& Tensor::at(std::size_t row, std::size_t col) {
    if (!ready() || row >= rows_ || col >= cols_) {
        throw std::out_of_range("tensor index out of range");
    }
    return storage_->at(offset_ + row * row_stride_ + col * col_stride_);
}

float Tensor::at(std::size_t row, std::size_t col) const {
    if (!ready() || row >= rows_ || col >= cols_) {
        throw std::out_of_range("tensor index out of range");
    }
    return storage_->at(offset_ + row * row_stride_ + col * col_stride_);
}

bool Tensor::ready() const {
    if (!storage_ || rows_ == 0 || cols_ == 0) {
        return false;
    }
    const std::size_t last_index = offset_ + (rows_ - 1) * row_stride_ + (cols_ - 1) * col_stride_;
    return last_index < storage_->size();
}

bool Tensor::contiguous() const {
    return ready() && offset_ == 0 && col_stride_ == 1 && row_stride_ == cols_;
}

bool Tensor::shares_storage_with(const Tensor& other) const {
    return storage_ && other.storage_ && storage_ == other.storage_;
}

std::size_t Tensor::storage_use_count() const {
    return storage_ ? storage_.use_count() : 0;
}

Tensor Tensor::reshape(std::size_t rows, std::size_t cols) const {
    if (!ready() || !contiguous() || rows * cols != rows_ * cols_) {
        return {};
    }
    return Tensor(storage_, rows, cols, offset_, cols, 1);
}

Tensor Tensor::slice(std::size_t row_start, std::size_t row_count, std::size_t col_start, std::size_t col_count) const {
    if (!ready() || row_start > rows_ || col_start > cols_) {
        return {};
    }
    if (row_count == 0 || col_count == 0) {
        return {};
    }
    if (row_start + row_count > rows_ || col_start + col_count > cols_) {
        return {};
    }

    const std::size_t new_offset = offset_ + row_start * row_stride_ + col_start * col_stride_;
    return Tensor(storage_, row_count, col_count, new_offset, row_stride_, col_stride_);
}

std::size_t Tensor::rows() const {
    return rows_;
}

std::size_t Tensor::cols() const {
    return cols_;
}

}  // namespace chapter02_tensor_ops
