// 多模态 CPU：SmolVLM 单轮图文 prompt 与 IDEFICS3 chunk 契约。
// 职责：把问题和 tile 网格展开为交替的文本/视觉片段，不执行 BPE 或视觉编码。
// 验证：multimodal_smolvlm_prompt_test 与 llama.cpp mtmd debug 日志。
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace multimodal {

enum class PromptChunkKind {
    Text,
    Visual,
};

struct PromptChunk {
    PromptChunkKind kind = PromptChunkKind::Text;
    std::string text;
    std::size_t visual_index = 0;
};

struct SmolvlmPromptChunks {
    std::vector<PromptChunk> chunks;
    std::size_t tile_count = 0;
    std::size_t visual_count = 0;
};

bool build_smolvlm_single_turn_prompt(
    const std::string& question,
    std::size_t tile_rows,
    std::size_t tile_cols,
    SmolvlmPromptChunks* output,
    std::string* error = nullptr);

}  // namespace multimodal
