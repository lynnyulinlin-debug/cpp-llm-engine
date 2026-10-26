// Chapter 05：实现 token-step path 使用的教学 DecoderLayer。
// 职责：保留 Chapter 04 结构边界，供 Runtime 组合调用。
// 验证：Chapter 05 generation 测试检查集成行为，不作为生产精度证明。
#include "chapter05_generation/decoder_layer.h"

namespace chapter05_generation {

DecoderLayer::DecoderLayer(std::size_t layer_index)
    : layer_index_(layer_index) {}

std::size_t DecoderLayer::layer_index() const {
    return layer_index_;
}

bool DecoderLayer::ready() const {
    return true;
}

Logits DecoderLayer::forward(const Logits& hidden) const {
    Logits output = hidden;
    const float bias = static_cast<float>(layer_index_) + 1.0f;
    for (float& value : output) {
        value += bias;
    }
    return output;
}

}  // namespace chapter05_generation
