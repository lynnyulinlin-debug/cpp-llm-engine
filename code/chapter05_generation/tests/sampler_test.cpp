// Chapter 05：独立于 model execution 测试 logits-to-token sampler behavior。
// 覆盖：保持 deterministic greedy 行为，并让 configurable sampling regression 可见。
// 验证入口：Chapter 05 CTest target 或 scripts/run.sh。
#include "chapter05_generation/sampler.h"

#include <cassert>

int main() {
    chapter05_generation::GreedySampler sampler;
    const chapter05_generation::TokenId token = sampler.sample({0.1f, 0.9f, 0.2f});
    assert(token == 1);
    return 0;
}
