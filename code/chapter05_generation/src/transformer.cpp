// Chapter 05：实现 CPU Generation Runtime 使用的教学 multi-layer Transformer。
// 职责：将 token representation 经过 position encoding 和 decoder layer 送到 logits。
// 验证：generation 测试检查 assembled prefill 和 decode path。
#include "chapter05_generation/transformer.h"

#include <utility>

namespace chapter05_generation {

Transformer::Transformer(
    std::size_t layer_count,
    Embedding embedding,
    PositionEncoding position_encoding)
    : layers_(),
      embedding_(std::move(embedding)),
      position_encoding_(std::move(position_encoding)) {
    layers_.reserve(layer_count);
    for (std::size_t i = 0; i < layer_count; ++i) {
        layers_.emplace_back(i);
    }
}

std::size_t Transformer::layer_count() const {
    return layers_.size();
}

const Embedding& Transformer::embedding() const {
    return embedding_;
}

const PositionEncoding& Transformer::position_encoding() const {
    return position_encoding_;
}

bool Transformer::ready() const {
    return !layers_.empty() && embedding_.ready() && position_encoding_.ready();
}

Logits Transformer::prefill(const Request& request) const {
    return forward(request, 0);
}

Logits Transformer::decode(const Request& request) const {
    return forward(request, request.generated_size());
}

Logits Transformer::forward(const Request& request, std::size_t step) const {
    TokenSequence tokens = request.prompt();
    const TokenSequence& generated = request.generated();
    tokens.insert(tokens.end(), generated.begin(), generated.end());

    Logits hidden = embedding_.forward(tokens);
    position_encoding_.apply(hidden, step);

    for (const DecoderLayer& layer : layers_) {
        hidden = layer.forward(hidden);
    }

    Logits logits;
    logits.reserve(3);
    const float base = hidden.empty() ? 0.0f : hidden.front();
    const float prompt_factor = static_cast<float>(request.prompt_size());
    const float generated_factor = static_cast<float>(request.generated_size());
    logits.push_back(base + prompt_factor);
    logits.push_back(base + generated_factor + 1.0f);
    logits.push_back(base + static_cast<float>(layers_.size()) + 2.0f);
    return logits;
}

}  // namespace chapter05_generation
