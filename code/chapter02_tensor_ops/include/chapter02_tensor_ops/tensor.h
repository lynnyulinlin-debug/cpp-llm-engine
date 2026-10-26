// Chapter 02：Tensor view 接口
// 职责：声明共享 storage 的 reshape、slice、stride 和生命周期观察接口。
// 契约：view 不复制数据；非法 shape、offset 或索引返回无效 view 或抛出边界异常。
// 验证：chapter02_tensor_ops_test.cpp。
#pragma once

#include "cpp_llm_engine/core/tensor_contract.h"

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
    cpp_llm_engine::core::Status try_reshape(
        std::size_t rows, std::size_t cols, Tensor* output) const;
    cpp_llm_engine::core::Status try_slice(
        std::size_t row_start,
        std::size_t row_count,
        std::size_t col_start,
        std::size_t col_count,
        Tensor* output) const;
    cpp_llm_engine::core::TensorContract contract() const;

    std::size_t rows() const;
    std::size_t cols() const;
    std::size_t offset() const;
    std::size_t row_stride() const;
    std::size_t col_stride() const;

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
