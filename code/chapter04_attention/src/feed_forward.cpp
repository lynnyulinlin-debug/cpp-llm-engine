#include "chapter04_attention/feed_forward.h"

namespace chapter04_attention {

FeedForward::FeedForward(std::size_t model_dim, std::size_t hidden_dim)
    : model_dim_(model_dim),
      hidden_dim_(hidden_dim),
      fc1_(model_dim, hidden_dim),
      fc2_(hidden_dim, model_dim) {}

Tensor FeedForward::forward(const Tensor& input) const {
    if (!ready() || !input.ready() || input.cols != model_dim_) {
        return {};
    }

    Tensor hidden = fc1_.forward(input);
    Tensor activated = apply_activation(hidden, ActivationKind::Relu);
    return fc2_.forward(activated);
}

bool FeedForward::ready() const {
    return model_dim_ > 0 && hidden_dim_ > 0 && fc1_.ready() && fc2_.ready();
}

std::size_t FeedForward::model_dim() const {
    return model_dim_;
}

std::size_t FeedForward::hidden_dim() const {
    return hidden_dim_;
}

Tensor apply_activation(const Tensor& input, ActivationKind kind) {
    if (!input.ready()) {
        return {};
    }

    Tensor out(input.rows, input.cols);
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        switch (kind) {
            case ActivationKind::Relu:
            default:
                out.data[i] = input.data[i] > 0.0f ? input.data[i] : 0.0f;
                break;
        }
    }
    return out;
}

}  // namespace chapter04_attention
