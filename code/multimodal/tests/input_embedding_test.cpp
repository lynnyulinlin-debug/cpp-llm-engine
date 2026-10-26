// 多模态 CPU：图文 embedding 拼接契约测试
// 职责：验证占位符展开、多个图片顺序、来源映射、shape 与错误输入。
// 契约：测试使用小型 Float32 矩阵，不依赖模型、Python、网络、llama.cpp 或 GPU。
// 运行：由 CTest 执行 multimodal_input_embedding_test。
#include "multimodal/input_embedding.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                                   \
    do {                                                                                   \
        if (!(condition)) {                                                                \
            std::cerr << "CHECK failed at line " << __LINE__ << ": " #condition << '\n'; \
            ++failures;                                                                    \
        }                                                                                  \
    } while (false)

}  // namespace

int main() {
    const multimodal::FloatMatrix text{
        5, 2, {10, 11, 20, 21, 30, 31, 40, 41, 50, 51}};
    const std::vector<multimodal::VisualEmbeddingInsertion> insertions{
        {3, {1, 2, {300, 301}}},
        {1, {2, 2, {100, 101, 110, 111}}},
    };
    multimodal::MultimodalEmbeddingSequence output;
    std::string error;
    CHECK(multimodal::compose_multimodal_embeddings(text, insertions, &output, &error));
    CHECK(error.empty());
    CHECK(output.embeddings.rows == 6 && output.embeddings.cols == 2);
    const std::vector<float> expected{
        10, 11, 100, 101, 110, 111, 30, 31, 300, 301, 50, 51};
    CHECK(output.embeddings.values == expected);
    CHECK(output.origins.size() == 6);
    CHECK(output.origins[0].kind == multimodal::EmbeddingOriginKind::Text);
    CHECK(output.origins[0].source_index == 0);
    CHECK(output.origins[1].kind == multimodal::EmbeddingOriginKind::Visual);
    CHECK(output.origins[1].source_index == 1 && output.origins[1].source_row == 0);
    CHECK(output.origins[2].source_index == 1 && output.origins[2].source_row == 1);
    CHECK(output.origins[3].kind == multimodal::EmbeddingOriginKind::Text);
    CHECK(output.origins[3].source_index == 2);
    CHECK(output.origins[4].kind == multimodal::EmbeddingOriginKind::Visual);
    CHECK(output.origins[4].source_index == 0 && output.origins[4].source_row == 0);
    CHECK(output.origins[5].source_index == 4);

    CHECK(multimodal::compose_multimodal_embeddings(text, {}, &output, &error));
    CHECK(output.embeddings.values == text.values);
    CHECK(output.origins.size() == text.rows);

    auto invalid_width = insertions;
    invalid_width[0].embeddings = {1, 3, {1, 2, 3}};
    CHECK(!multimodal::compose_multimodal_embeddings(text, invalid_width, &output, &error));
    CHECK(error.find("width") != std::string::npos);
    CHECK(!multimodal::compose_multimodal_embeddings(
        text, {{5, {1, 2, {1, 2}}}}, &output, &error));
    CHECK(error.find("outside") != std::string::npos);
    CHECK(!multimodal::compose_multimodal_embeddings(
        text, {{1, {1, 2, {1, 2}}}, {1, {1, 2, {3, 4}}}}, &output, &error));
    CHECK(error.find("duplicated") != std::string::npos);
    CHECK(!multimodal::compose_multimodal_embeddings(text, insertions, nullptr, &error));

    if (failures != 0) return 1;
    std::cout << "multimodal input embedding checks passed\n";
    return 0;
}
