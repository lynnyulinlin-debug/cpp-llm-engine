#pragma once

#include "chapter05_generation/generation_types.h"

#include <vector>

namespace chapter05_generation {

class Sampler {
public:
    virtual ~Sampler() = default;
    virtual TokenId sample(const std::vector<float>& logits) const = 0;
};

class GreedySampler final : public Sampler {
public:
    TokenId sample(const std::vector<float>& logits) const override;
};

}  // namespace chapter05_generation
