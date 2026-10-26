// Chapter 05：实现教学 Transformer 的确定性 position encoding。
// 职责：显式表达 prefill 和 decode step 的 position progression。
// 验证：Runtime 测试检查最终 sequence flow。
#include "chapter05_generation/position_encoding.h"

namespace chapter05_generation {

PositionEncoding::PositionEncoding(std::size_t max_length)
    : max_length_(max_length) {}

std::size_t PositionEncoding::max_length() const {
    return max_length_;
}

bool PositionEncoding::ready() const {
    return max_length_ > 0;
}

void PositionEncoding::apply(Logits& hidden, std::size_t position_base) const {
    if (hidden.empty()) {
        return;
    }

    for (std::size_t i = 0; i < hidden.size(); ++i) {
        const std::size_t position = position_base + i;
        if (max_length_ == 0) {
            hidden[i] += static_cast<float>(position);
        } else {
            hidden[i] += static_cast<float>(position % max_length_);
        }
    }
}

}  // namespace chapter05_generation
