// Chapter 05：声明 prefill 和 decode 使用的 token id -> embedding lookup。
// 契约：持有词表大小的 embedding table，返回确定性的 CPU 向量。
// 验证：generation 和 runtime smoke 测试检查 lookup 边界与输出 shape。
#pragma once

#include "chapter05_generation/generation_types.h"

#include <cstddef>

namespace chapter05_generation {

class Embedding {
public:
    Embedding(std::size_t vocab_size = 0);

    std::size_t vocab_size() const;
    bool ready() const;
    Logits forward(const TokenSequence& tokens) const;

private:
    std::size_t vocab_size_ = 0;
};

}  // namespace chapter05_generation
