// Chapter 02：Tensor view 实现
// 职责：实现共享 storage、offset/stride 索引、reshape 和 slice。
// 契约：所有 shape/index 乘加经过溢出检查，连续 view 才允许 reshape。
// 验证：code/chapter02_tensor_ops/scripts/run.sh。
#include "chapter02_tensor_ops/tensor.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace chapter02_tensor_ops {

namespace {

bool checked_multiply(std::size_t lhs, std::size_t rhs, std::size_t& result) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t& result) {
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool checked_index(std::size_t offset,
                   std::size_t row,
                   std::size_t row_stride,
                   std::size_t col,
                   std::size_t col_stride,
                   std::size_t& result) {
    std::size_t row_offset = 0;
    std::size_t col_offset = 0;
    if (!checked_multiply(row, row_stride, row_offset) ||
        !checked_multiply(col, col_stride, col_offset) ||
        !checked_add(offset, row_offset, result) ||
        !checked_add(result, col_offset, result)) {
        return false;
    }
    return true;
}

}  // namespace

Tensor::Tensor() = default;

Tensor::Tensor(std::size_t rows, std::size_t cols, float value)
    : rows_(rows),
      cols_(cols),
      offset_(0),
      row_stride_(cols),
      col_stride_(1) {
    std::size_t element_count = 0;
    if (!checked_multiply(rows, cols, element_count)) {
        throw std::length_error("tensor shape is too large");
    }
    storage_ = std::make_shared<std::vector<float>>(element_count, value);
}

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
    std::size_t index = 0;
    if (!checked_index(offset_, row, row_stride_, col, col_stride_, index)) {
        throw std::out_of_range("tensor index overflow");
    }
    return storage_->at(index);
}

float Tensor::at(std::size_t row, std::size_t col) const {
    if (!ready() || row >= rows_ || col >= cols_) {
        throw std::out_of_range("tensor index out of range");
    }
    std::size_t index = 0;
    if (!checked_index(offset_, row, row_stride_, col, col_stride_, index)) {
        throw std::out_of_range("tensor index overflow");
    }
    return storage_->at(index);
}

bool Tensor::ready() const {
    if (!storage_ || rows_ == 0 || cols_ == 0) {
        return false;
    }
    std::size_t last_index = 0;
    return checked_index(offset_, rows_ - 1, row_stride_, cols_ - 1, col_stride_, last_index) &&
           last_index < storage_->size();
}

bool Tensor::contiguous() const {
    // A one-row view is contiguous even when its parent has a larger row stride.
    return ready() && col_stride_ == 1 && (rows_ == 1 || row_stride_ == cols_);
}

bool Tensor::shares_storage_with(const Tensor& other) const {
    return storage_ && other.storage_ && storage_ == other.storage_;
}

std::size_t Tensor::storage_use_count() const {
    return storage_ ? storage_.use_count() : 0;
}

Tensor Tensor::reshape(std::size_t rows, std::size_t cols) const {
    Tensor output;
    if (!try_reshape(rows, cols, &output).is_ok()) return {};
    return output;
}

cpp_llm_engine::core::Status Tensor::try_reshape(
    std::size_t rows,
    std::size_t cols,
    Tensor* output) const {
    if (output == nullptr) {
        return cpp_llm_engine::core::Status::error("reshape output must not be null");
    }
    *output = {};
    std::size_t old_element_count = 0;
    std::size_t new_element_count = 0;
    if (!ready()) {
        return cpp_llm_engine::core::Status::error("reshape source tensor is not ready");
    }
    if (!contiguous()) {
        return cpp_llm_engine::core::Status::error("reshape requires a contiguous tensor");
    }
    if (rows == 0 || cols == 0) {
        return cpp_llm_engine::core::Status::error("reshape dimensions must be non-zero");
    }
    if (!checked_multiply(rows_, cols_, old_element_count) ||
        !checked_multiply(rows, cols, new_element_count)) {
        return cpp_llm_engine::core::Status::error("reshape element count overflows host size");
    }
    if (new_element_count != old_element_count) {
        return cpp_llm_engine::core::Status::error("reshape element count must not change");
    }
    *output = Tensor(storage_, rows, cols, offset_, cols, 1);
    return cpp_llm_engine::core::Status::ok();
}

Tensor Tensor::slice(std::size_t row_start, std::size_t row_count, std::size_t col_start, std::size_t col_count) const {
    Tensor output;
    if (!try_slice(row_start, row_count, col_start, col_count, &output).is_ok()) {
        return {};
    }
    return output;
}

cpp_llm_engine::core::Status Tensor::try_slice(
    std::size_t row_start,
    std::size_t row_count,
    std::size_t col_start,
    std::size_t col_count,
    Tensor* output) const {
    if (output == nullptr) {
        return cpp_llm_engine::core::Status::error("slice output must not be null");
    }
    *output = {};
    if (!ready()) {
        return cpp_llm_engine::core::Status::error("slice source tensor is not ready");
    }
    if (row_start > rows_ || col_start > cols_) {
        return cpp_llm_engine::core::Status::error("slice start is outside tensor shape");
    }
    if (row_count == 0 || col_count == 0) {
        return cpp_llm_engine::core::Status::error("slice dimensions must be non-zero");
    }
    if (row_count > rows_ - row_start || col_count > cols_ - col_start) {
        return cpp_llm_engine::core::Status::error("slice range exceeds tensor shape");
    }

    std::size_t new_offset = 0;
    if (!checked_index(offset_, row_start, row_stride_, col_start, col_stride_, new_offset)) {
        return cpp_llm_engine::core::Status::error("slice offset overflows host size");
    }

    *output = Tensor(storage_, row_count, col_count, new_offset, row_stride_, col_stride_);
    return cpp_llm_engine::core::Status::ok();
}

cpp_llm_engine::core::TensorContract Tensor::contract() const {
    return cpp_llm_engine::core::TensorContract{
        cpp_llm_engine::core::Shape{{rows_, cols_}},
        {row_stride_, col_stride_},
        cpp_llm_engine::core::DType::Float32,
        cpp_llm_engine::core::DeviceKind::CPU,
        offset_,
        storage_ ? storage_->size() * sizeof(float) : 0,
    };
}

std::size_t Tensor::rows() const {
    return rows_;
}

std::size_t Tensor::cols() const {
    return cols_;
}

std::size_t Tensor::offset() const {
    return offset_;
}

std::size_t Tensor::row_stride() const {
    return row_stride_;
}

std::size_t Tensor::col_stride() const {
    return col_stride_;
}

}  // namespace chapter02_tensor_ops
