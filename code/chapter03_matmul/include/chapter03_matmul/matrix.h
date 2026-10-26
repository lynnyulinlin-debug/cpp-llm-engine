// Chapter 03：教学 Matrix 对象
// 职责：保存 row-major float 矩阵，提供安全索引和 ready 检查。
// 契约：data.size() 必须等于 rows * cols，shape/index 溢出或越界会失败。
// 验证：chapter03_matmul_test.cpp。
#pragma once

#include "cpp_llm_engine/core/tensor_contract.h"

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
    cpp_llm_engine::core::TensorContract contract() const;
};

}  // namespace chapter03_matmul
