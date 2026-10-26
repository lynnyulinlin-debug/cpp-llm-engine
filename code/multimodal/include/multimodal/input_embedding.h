// 多模态 CPU：图文输入 embedding 序列接口
// 职责：用 visual embedding 替换文本序列中的图片占位行，并保留输出来源映射。
// 契约：文本与视觉矩阵 width 必须一致；每个占位位置只能被替换一次。
// 验证：multimodal_input_embedding_test 与真实 SmolVLM input inspect。
#pragma once

#include "multimodal/vision_encoder.h"

#include <cstddef>
#include <string>
#include <vector>

namespace multimodal {

enum class EmbeddingOriginKind {
    Text,
    Visual,
};

struct EmbeddingOrigin {
    EmbeddingOriginKind kind = EmbeddingOriginKind::Text;
    std::size_t source_index = 0;
    std::size_t source_row = 0;
};

struct VisualEmbeddingInsertion {
    std::size_t placeholder_position = 0;
    FloatMatrix embeddings;
};

struct MultimodalEmbeddingSequence {
    FloatMatrix embeddings;
    std::vector<EmbeddingOrigin> origins;
};

bool compose_multimodal_embeddings(
    const FloatMatrix& text_embeddings,
    const std::vector<VisualEmbeddingInsertion>& visual_insertions,
    MultimodalEmbeddingSequence* output,
    std::string* error = nullptr);

}  // namespace multimodal
