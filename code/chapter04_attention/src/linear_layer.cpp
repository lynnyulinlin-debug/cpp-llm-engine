#include "chapter04_attention/linear_layer.h"

#include <stdexcept>

namespace chapter04_attention {

namespace {

Tensor make_weight(std::size_t in_features, std::size_t out_features) {
    Tensor weight(in_features, out_features);
    for (std::size_t row = 0; row < in_features; ++row) {
        for (std::size_t col = 0; col < out_features; ++col) {
            weight.at(row, col) = 0.01f * static_cast<float>((row + 1) * (col + 1));
        }
    }
    return weight;
}

}  // namespace

LinearLayer::LinearLayer(std::size_t in_features, std::size_t out_features)
    : in_features_(in_features),
      out_features_(out_features),
      weight_(make_weight(in_features, out_features)),
      bias_(out_features, 0.0f) {}

Tensor LinearLayer::forward(const Tensor& input) const {
    if (!ready() || !input.ready() || input.cols != in_features_) {
        return {};
    }

    Tensor out = matmul(input, weight_);
    for (std::size_t row = 0; row < out.rows; ++row) {
        for (std::size_t col = 0; col < out.cols; ++col) {
            out.at(row, col) += bias_[col];
        }
    }
    return out;
}

bool LinearLayer::ready() const {
    return in_features_ > 0 && out_features_ > 0 &&
           weight_.ready() && weight_.rows == in_features_ && weight_.cols == out_features_ &&
           bias_.size() == out_features_;
}

std::size_t LinearLayer::in_features() const {
    return in_features_;
}

std::size_t LinearLayer::out_features() const {
    return out_features_;
}

}  // namespace chapter04_attention
