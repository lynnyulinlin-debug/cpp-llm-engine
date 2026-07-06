#pragma once

#include "chapter05_generation/generation_types.h"

#include <cstddef>

namespace chapter05_generation {

class PositionEncoding {
public:
    PositionEncoding(std::size_t max_length = 0);

    std::size_t max_length() const;
    bool ready() const;
    void apply(Logits& hidden, std::size_t position_base) const;

private:
    std::size_t max_length_ = 0;
};

}  // namespace chapter05_generation
