// Chapter 04：Decoder Tensor 和辅助算子实现
// 职责：实现 add、transpose、matmul、列切片、拼接和缩放。
// 契约：操作失败返回空 Tensor，所有结果使用独立 row-major storage。
// 验证：chapter04_attention_test.cpp。
#include "chapter04_attention/tensor.h"

#include <algorithm>
#include <stdexcept>

namespace chapter04_attention {

Tensor::Tensor(std::size_t r, std::size_t c, float value)
    : rows(r), cols(c), data(r * c, value) {}

float& Tensor::at(std::size_t row, std::size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("tensor index out of range");
    }
    return data[row * cols + col];
}

float Tensor::at(std::size_t row, std::size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("tensor index out of range");
    }
    return data[row * cols + col];
}

bool Tensor::ready() const {
    return rows > 0 && cols > 0 && data.size() == rows * cols;
}

Tensor add(const Tensor& lhs, const Tensor& rhs) {
    if (!lhs.ready() || !rhs.ready() || lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
        return {};
    }

    Tensor out(lhs.rows, lhs.cols);
    for (std::size_t i = 0; i < out.data.size(); ++i) {
        out.data[i] = lhs.data[i] + rhs.data[i];
    }
    return out;
}

Tensor transpose(const Tensor& tensor) {
    if (!tensor.ready()) {
        return {};
    }

    Tensor out(tensor.cols, tensor.rows);
    for (std::size_t row = 0; row < tensor.rows; ++row) {
        for (std::size_t col = 0; col < tensor.cols; ++col) {
            out.at(col, row) = tensor.at(row, col);
        }
    }
    return out;
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    if (!lhs.ready() || !rhs.ready() || lhs.cols != rhs.rows) {
        return {};
    }

    Tensor out(lhs.rows, rhs.cols);
    for (std::size_t row = 0; row < lhs.rows; ++row) {
        for (std::size_t col = 0; col < rhs.cols; ++col) {
            float value = 0.0f;
            for (std::size_t k = 0; k < lhs.cols; ++k) {
                value += lhs.at(row, k) * rhs.at(k, col);
            }
            out.at(row, col) = value;
        }
    }
    return out;
}

Tensor slice_columns(const Tensor& tensor, std::size_t start_col, std::size_t width) {
    if (!tensor.ready() || start_col > tensor.cols || width > tensor.cols - start_col) {
        return {};
    }

    Tensor out(tensor.rows, width);
    for (std::size_t row = 0; row < tensor.rows; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
            out.at(row, col) = tensor.at(row, start_col + col);
        }
    }
    return out;
}

Tensor concatenate_columns(const std::vector<Tensor>& parts) {
    if (parts.empty()) {
        return {};
    }

    const std::size_t rows = parts.front().rows;
    std::size_t cols = 0;
    for (const auto& part : parts) {
        if (!part.ready() || part.rows != rows) {
            return {};
        }
        cols += part.cols;
    }

    Tensor out(rows, cols);
    std::size_t offset = 0;
    for (const auto& part : parts) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < part.cols; ++col) {
                out.at(row, offset + col) = part.at(row, col);
            }
        }
        offset += part.cols;
    }
    return out;
}

void scale_in_place(Tensor& tensor, float factor) {
    if (!tensor.ready()) {
        return;
    }
    for (float& value : tensor.data) {
        value *= factor;
    }
}

}  // namespace chapter04_attention
