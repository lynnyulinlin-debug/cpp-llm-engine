// Chapter 04：LayerNorm 接口
// 职责：声明按 token 行执行的 LayerNorm、gamma、beta 和 epsilon 参数。
// 契约：输入列数必须等于 dim，输出 shape 与输入一致。
// 验证：chapter04_attention_test.cpp。
#pragma once

#include "chapter04_attention/tensor.h"

#include <cstddef>
#include <vector>

namespace chapter04_attention {

class LayerNorm final {
public:
    explicit LayerNorm(std::size_t dim = 0, float epsilon = 1e-5f);

    Tensor forward(const Tensor& input) const;
    bool ready() const;

    std::size_t dim() const;
    float epsilon() const;

private:
    std::size_t dim_ = 0;
    float epsilon_ = 1e-5f;
    std::vector<float> gamma_;
    std::vector<float> beta_;
};

}  // namespace chapter04_attention
