#include "chapter04_attention/layer_norm.h"

#include <cmath>

namespace chapter04_attention {

LayerNorm::LayerNorm(std::size_t dim, float epsilon)
    : dim_(dim), epsilon_(epsilon), gamma_(dim, 1.0f), beta_(dim, 0.0f) {}

Tensor LayerNorm::forward(const Tensor& input) const {
    if (!ready() || !input.ready() || input.cols != dim_) {
        return {};
    }

    Tensor out(input.rows, input.cols);
    for (std::size_t row = 0; row < input.rows; ++row) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < input.cols; ++col) {
            sum += input.at(row, col);
        }
        const float mean = sum / static_cast<float>(input.cols);

        float var_sum = 0.0f;
        for (std::size_t col = 0; col < input.cols; ++col) {
            const float diff = input.at(row, col) - mean;
            var_sum += diff * diff;
        }
        const float variance = var_sum / static_cast<float>(input.cols);
        const float denom = std::sqrt(variance + epsilon_);

        for (std::size_t col = 0; col < input.cols; ++col) {
            const float normalized = (input.at(row, col) - mean) / denom;
            out.at(row, col) = normalized * gamma_[col] + beta_[col];
        }
    }
    return out;
}

bool LayerNorm::ready() const {
    return dim_ > 0 && gamma_.size() == dim_ && beta_.size() == dim_;
}

std::size_t LayerNorm::dim() const {
    return dim_;
}

float LayerNorm::epsilon() const {
    return epsilon_;
}

}  // namespace chapter04_attention
