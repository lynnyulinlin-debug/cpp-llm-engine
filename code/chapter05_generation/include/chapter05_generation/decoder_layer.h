// Chapter 05：Generation Core 使用的教学 DecoderLayer 声明。
// 职责：连接 Chapter 04 的 decoder contract 和 Chapter 05 的 token-step 接口。
// 验证：Runtime 和 decoder 测试检查 shape 流转与输出推进。
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
