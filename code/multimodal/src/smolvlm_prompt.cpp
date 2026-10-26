// 多模态 CPU：实现 SmolVLM 单轮模板和 IDEFICS3 图片边界片段。
// 契约：tile 按行优先排列，overview 位于最后，文本与视觉片段严格交替。
// 验证：multimodal_smolvlm_prompt_test 覆盖 overview-only、2x3 网格和错误输入。
#include "multimodal/smolvlm_prompt.h"

#include <limits>
#include <utility>

namespace multimodal {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

void append_text(std::vector<PromptChunk>* chunks, std::string text) {
    if (text.empty()) return;
    if (!chunks->empty() && chunks->back().kind == PromptChunkKind::Text) {
        chunks->back().text += text;
        return;
    }
    chunks->push_back({PromptChunkKind::Text, std::move(text), 0});
}

void append_visual(std::vector<PromptChunk>* chunks, std::size_t visual_index) {
    chunks->push_back({PromptChunkKind::Visual, {}, visual_index});
}

}  // namespace

bool build_smolvlm_single_turn_prompt(
    const std::string& question,
    std::size_t tile_rows,
    std::size_t tile_cols,
    SmolvlmPromptChunks* output,
    std::string* error) {
    if (output == nullptr) {
        set_error(error, "prompt output must not be null");
        return false;
    }
    if (question.empty()) {
        set_error(error, "prompt question must not be empty");
        return false;
    }
    if ((tile_rows == 0) != (tile_cols == 0)) {
        set_error(error, "tile rows and columns must both be zero or both be positive");
        return false;
    }
    if (tile_rows != 0 && tile_cols > std::numeric_limits<std::size_t>::max() / tile_rows) {
        set_error(error, "tile grid size overflows");
        return false;
    }

    SmolvlmPromptChunks result;
    result.tile_count = tile_rows * tile_cols;
    result.visual_count = result.tile_count + 1;
    append_text(&result.chunks, "<|im_start|>User: ");

    std::size_t visual_index = 0;
    for (std::size_t row = 0; row < tile_rows; ++row) {
        for (std::size_t col = 0; col < tile_cols; ++col) {
            append_text(
                &result.chunks,
                "<fake_token_around_image><row_" + std::to_string(row + 1) +
                    "_col_" + std::to_string(col + 1) + ">");
            append_visual(&result.chunks, visual_index++);
        }
        if (row + 1 < tile_rows) append_text(&result.chunks, "\n");
    }

    append_text(&result.chunks, "\n\n<fake_token_around_image><global-img>");
    append_visual(&result.chunks, visual_index);
    append_text(
        &result.chunks,
        "<fake_token_around_image>" + question +
            "<end_of_utterance>\nAssistant:");

    *output = std::move(result);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace multimodal
