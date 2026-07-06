#include "chapter04_attention/attention.h"

#include <cmath>
#include <vector>

namespace chapter04_attention {

namespace {

Tensor softmax_rows(const Tensor& tensor) {
    if (!tensor.ready()) {
        return {};
    }

    Tensor out(tensor.rows, tensor.cols);
    for (std::size_t row = 0; row < tensor.rows; ++row) {
        float max_value = tensor.at(row, 0);
        for (std::size_t col = 1; col < tensor.cols; ++col) {
            max_value = std::max(max_value, tensor.at(row, col));
        }

        float sum = 0.0f;
        for (std::size_t col = 0; col < tensor.cols; ++col) {
            const float value = std::exp(tensor.at(row, col) - max_value);
            out.at(row, col) = value;
            sum += value;
        }
        if (sum == 0.0f) {
            continue;
        }
        for (std::size_t col = 0; col < tensor.cols; ++col) {
            out.at(row, col) /= sum;
        }
    }
    return out;
}

}  // namespace

MultiHeadAttention::MultiHeadAttention(std::size_t model_dim, std::size_t num_heads)
    : model_dim_(model_dim),
      num_heads_(num_heads),
      q_proj_(model_dim, model_dim),
      k_proj_(model_dim, model_dim),
      v_proj_(model_dim, model_dim),
      out_proj_(model_dim, model_dim) {}

Tensor MultiHeadAttention::forward(const Tensor& input) const {
    if (!ready() || !input.ready() || input.cols != model_dim_) {
        return {};
    }

    const std::size_t head_dim = model_dim_ / num_heads_;
    Tensor q = q_proj_.forward(input);
    Tensor k = k_proj_.forward(input);
    Tensor v = v_proj_.forward(input);

    std::vector<Tensor> head_outputs;
    head_outputs.reserve(num_heads_);

    for (std::size_t head = 0; head < num_heads_; ++head) {
        const std::size_t offset = head * head_dim;
        Tensor q_head = slice_columns(q, offset, head_dim);
        Tensor k_head = slice_columns(k, offset, head_dim);
        Tensor v_head = slice_columns(v, offset, head_dim);

        Tensor scores = matmul(q_head, transpose(k_head));
        scale_in_place(scores, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        Tensor probs = softmax_rows(scores);
        Tensor context = matmul(probs, v_head);
        head_outputs.push_back(context);
    }

    Tensor combined = concatenate_columns(head_outputs);
    return out_proj_.forward(combined);
}

bool MultiHeadAttention::ready() const {
    return model_dim_ > 0 && num_heads_ > 0 && model_dim_ % num_heads_ == 0 &&
           q_proj_.ready() && k_proj_.ready() && v_proj_.ready() && out_proj_.ready();
}

std::size_t MultiHeadAttention::model_dim() const {
    return model_dim_;
}

std::size_t MultiHeadAttention::num_heads() const {
    return num_heads_;
}

}  // namespace chapter04_attention
