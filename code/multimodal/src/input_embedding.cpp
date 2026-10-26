// 多模态 CPU：图文输入 embedding 序列实现
// 职责：按文本顺序复制普通行，并在图片占位位置展开对应 visual embedding。
// 契约：占位行本身不进入结果；输出行数为 text - placeholder + visual rows。
// 验证：multimodal_input_embedding_test 与真实 SmolVLM input inspect。
#include "multimodal/input_embedding.h"

#include <limits>
#include <utility>

namespace multimodal {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t* result) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) return false;
    *result = lhs + rhs;
    return true;
}

}  // namespace

bool compose_multimodal_embeddings(
    const FloatMatrix& text_embeddings,
    const std::vector<VisualEmbeddingInsertion>& visual_insertions,
    MultimodalEmbeddingSequence* output,
    std::string* error) {
    if (output == nullptr) return fail(error, "multimodal embedding output must not be null");
    if (!text_embeddings.valid()) return fail(error, "text embedding matrix is invalid");

    std::vector<std::size_t> insertion_at(text_embeddings.rows,
                                          std::numeric_limits<std::size_t>::max());
    std::size_t output_rows = text_embeddings.rows;
    for (std::size_t index = 0; index < visual_insertions.size(); ++index) {
        const auto& insertion = visual_insertions[index];
        if (insertion.placeholder_position >= text_embeddings.rows) {
            return fail(error, "visual placeholder position is outside the text sequence");
        }
        if (insertion_at[insertion.placeholder_position] !=
            std::numeric_limits<std::size_t>::max()) {
            return fail(error, "visual placeholder position is duplicated");
        }
        if (!insertion.embeddings.valid() ||
            insertion.embeddings.cols != text_embeddings.cols) {
            return fail(error, "visual embedding matrix does not match text width");
        }
        insertion_at[insertion.placeholder_position] = index;
        --output_rows;
        if (!checked_add(output_rows, insertion.embeddings.rows, &output_rows)) {
            return fail(error, "multimodal sequence row count overflows");
        }
    }
    if (output_rows > std::numeric_limits<std::size_t>::max() / text_embeddings.cols) {
        return fail(error, "multimodal sequence element count overflows");
    }

    MultimodalEmbeddingSequence result;
    result.embeddings.rows = output_rows;
    result.embeddings.cols = text_embeddings.cols;
    result.embeddings.values.reserve(output_rows * text_embeddings.cols);
    result.origins.reserve(output_rows);
    for (std::size_t text_row = 0; text_row < text_embeddings.rows; ++text_row) {
        const std::size_t insertion_index = insertion_at[text_row];
        if (insertion_index == std::numeric_limits<std::size_t>::max()) {
            const auto begin = text_embeddings.values.begin() + text_row * text_embeddings.cols;
            result.embeddings.values.insert(
                result.embeddings.values.end(), begin, begin + text_embeddings.cols);
            result.origins.push_back({EmbeddingOriginKind::Text, text_row, 0});
            continue;
        }
        const auto& visual = visual_insertions[insertion_index].embeddings;
        result.embeddings.values.insert(
            result.embeddings.values.end(), visual.values.begin(), visual.values.end());
        for (std::size_t visual_row = 0; visual_row < visual.rows; ++visual_row) {
            result.origins.push_back(
                {EmbeddingOriginKind::Visual, insertion_index, visual_row});
        }
    }
    if (!result.embeddings.valid() || result.origins.size() != output_rows) {
        return fail(error, "multimodal sequence assembly produced an invalid result");
    }
    *output = std::move(result);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace multimodal
