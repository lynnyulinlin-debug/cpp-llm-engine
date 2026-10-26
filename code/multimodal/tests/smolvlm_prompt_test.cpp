// 多模态 CPU 测试：验证 SmolVLM 模板和 IDEFICS3 chunk 展开顺序。
// 覆盖：overview-only、2x3 tiles、行分隔、visual index、空问题和非法网格。
// 运行：ctest --test-dir <multimodal-build> --output-on-failure。
#include "multimodal/smolvlm_prompt.h"

#include <cassert>
#include <limits>
#include <string>

namespace {

using multimodal::PromptChunkKind;
using multimodal::SmolvlmPromptChunks;

void test_overview_only() {
    SmolvlmPromptChunks prompt;
    std::string error;
    assert(multimodal::build_smolvlm_single_turn_prompt("Describe it.", 0, 0, &prompt, &error));
    assert(error.empty());
    assert(prompt.tile_count == 0);
    assert(prompt.visual_count == 1);
    assert(prompt.chunks.size() == 3);
    assert(prompt.chunks[0].kind == PromptChunkKind::Text);
    assert(prompt.chunks[0].text ==
           "<|im_start|>User: \n\n<fake_token_around_image><global-img>");
    assert(prompt.chunks[1].kind == PromptChunkKind::Visual);
    assert(prompt.chunks[1].visual_index == 0);
    assert(prompt.chunks[2].text ==
           "<fake_token_around_image>Describe it.<end_of_utterance>\nAssistant:");
}

void test_two_by_three_tiles() {
    SmolvlmPromptChunks prompt;
    std::string error;
    assert(multimodal::build_smolvlm_single_turn_prompt("Question?", 2, 3, &prompt, &error));
    assert(prompt.tile_count == 6);
    assert(prompt.visual_count == 7);
    assert(prompt.chunks.size() == 15);
    for (std::size_t index = 0; index < prompt.visual_count; ++index) {
        const auto& visual = prompt.chunks[index * 2 + 1];
        assert(visual.kind == PromptChunkKind::Visual);
        assert(visual.visual_index == index);
    }
    assert(prompt.chunks[0].text ==
           "<|im_start|>User: <fake_token_around_image><row_1_col_1>");
    assert(prompt.chunks[4].text ==
           "<fake_token_around_image><row_1_col_3>");
    assert(prompt.chunks[6].text ==
           "\n<fake_token_around_image><row_2_col_1>");
    assert(prompt.chunks[10].text ==
           "<fake_token_around_image><row_2_col_3>");
    assert(prompt.chunks[12].text ==
           "\n\n<fake_token_around_image><global-img>");
    assert(prompt.chunks[14].text ==
           "<fake_token_around_image>Question?<end_of_utterance>\nAssistant:");
}

void test_invalid_inputs() {
    SmolvlmPromptChunks prompt;
    std::string error;
    assert(!multimodal::build_smolvlm_single_turn_prompt("", 0, 0, &prompt, &error));
    assert(!error.empty());
    assert(!multimodal::build_smolvlm_single_turn_prompt("x", 1, 0, &prompt, &error));
    assert(!multimodal::build_smolvlm_single_turn_prompt("x", 0, 1, &prompt, &error));
    assert(!multimodal::build_smolvlm_single_turn_prompt(
        "x", 2, std::numeric_limits<std::size_t>::max(), &prompt, &error));
    assert(!multimodal::build_smolvlm_single_turn_prompt("x", 0, 0, nullptr, &error));
}

}  // namespace

int main() {
    test_overview_only();
    test_two_by_three_tiles();
    test_invalid_inputs();
    return 0;
}
