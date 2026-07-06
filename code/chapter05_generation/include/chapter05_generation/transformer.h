#pragma once

#include "chapter05_generation/decoder_layer.h"
#include "chapter05_generation/embedding.h"
#include "chapter05_generation/position_encoding.h"
#include "chapter05_generation/request.h"

#include <cstddef>
#include <vector>

namespace chapter05_generation {

class Transformer {
public:
    Transformer(
        std::size_t layer_count = 0,
        Embedding embedding = {},
        PositionEncoding position_encoding = {});

    std::size_t layer_count() const;
    const Embedding& embedding() const;
    const PositionEncoding& position_encoding() const;
    bool ready() const;
    Logits prefill(const Request& request) const;
    Logits decode(const Request& request) const;

private:
    Logits forward(const Request& request, std::size_t step) const;

    std::vector<DecoderLayer> layers_;
    Embedding embedding_;
    PositionEncoding position_encoding_;
};

}  // namespace chapter05_generation
