#pragma once

#include "chapter05_generation/generation_types.h"

#include <cstddef>

namespace chapter05_generation {

class DecoderLayer {
public:
    explicit DecoderLayer(std::size_t layer_index = 0);

    std::size_t layer_index() const;
    bool ready() const;
    Logits forward(const Logits& hidden) const;

private:
    std::size_t layer_index_ = 0;
};

}  // namespace chapter05_generation
