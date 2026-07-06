#include "chapter05_generation/sampler.h"

#include <algorithm>

namespace chapter05_generation {

TokenId GreedySampler::sample(const std::vector<float>& logits) const {
    if (logits.empty()) {
        return 0;
    }

    return static_cast<TokenId>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

}  // namespace chapter05_generation
