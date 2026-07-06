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
