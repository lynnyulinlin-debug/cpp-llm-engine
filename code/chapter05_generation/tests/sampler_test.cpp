#include "chapter05_generation/sampler.h"

#include <cassert>

int main() {
    chapter05_generation::GreedySampler sampler;
    const chapter05_generation::TokenId token = sampler.sample({0.1f, 0.9f, 0.2f});
    assert(token == 1);
    return 0;
}
