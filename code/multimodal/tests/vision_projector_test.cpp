// 多模态 CPU：IDEFICS3 spatial merge 与 Projector 测试
// 职责：验证输出网格顺序、dy/dx/hidden 合并顺序、Linear 投影和错误输入。
// 契约：测试使用小矩阵，不依赖真实 mmproj、Python、网络、llama.cpp 或 GPU。
// 运行：由 CTest 执行 multimodal_vision_projector_test。
#include "multimodal/vision_projector.h"

#include <cmath>
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

bool near(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1.0e-5f;
}

}  // namespace

int main() {
    constexpr std::size_t grid_width = 8;
    constexpr std::size_t grid_height = 4;
    constexpr std::size_t hidden = 2;
    constexpr std::size_t merge = 4;
    multimodal::FloatMatrix input;
    input.rows = grid_width * grid_height;
    input.cols = hidden;
    input.values.resize(input.rows * input.cols);
    for (std::size_t token = 0; token < input.rows; ++token) {
        input.at(token, 0) = static_cast<float>(token * 10);
        input.at(token, 1) = static_cast<float>(token * 10 + 1);
    }

    multimodal::FloatMatrix merged;
    std::string error;
    CHECK(multimodal::merge_spatial_patches(
        input, grid_width, grid_height, merge, &merged, &error));
    CHECK(error.empty());
    CHECK(merged.rows == 2);
    CHECK(merged.cols == merge * merge * hidden);

    const std::vector<std::size_t> first_group{
        0, 1, 2, 3,
        8, 9, 10, 11,
        16, 17, 18, 19,
        24, 25, 26, 27};
    const std::vector<std::size_t> second_group{
        4, 5, 6, 7,
        12, 13, 14, 15,
        20, 21, 22, 23,
        28, 29, 30, 31};
    for (std::size_t i = 0; i < first_group.size(); ++i) {
        CHECK(near(merged.at(0, i * hidden), static_cast<float>(first_group[i] * 10)));
        CHECK(near(merged.at(0, i * hidden + 1), static_cast<float>(first_group[i] * 10 + 1)));
        CHECK(near(merged.at(1, i * hidden), static_cast<float>(second_group[i] * 10)));
        CHECK(near(merged.at(1, i * hidden + 1), static_cast<float>(second_group[i] * 10 + 1)));
    }

    multimodal::LinearWeights projector;
    projector.input_size = merged.cols;
    projector.output_size = 1;
    projector.weight.assign(merged.cols, 1.0f);
    multimodal::Idefics3ProjectorTrace trace;
    CHECK(multimodal::run_idefics3_projector(
        input, grid_width, grid_height, merge, projector, &trace, &error));
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    for (const std::size_t token : first_group) first_sum += token * 20.0f + 1.0f;
    for (const std::size_t token : second_group) second_sum += token * 20.0f + 1.0f;
    CHECK(trace.output.rows == 2 && trace.output.cols == 1);
    CHECK(near(trace.output.at(0, 0), first_sum));
    CHECK(near(trace.output.at(1, 0), second_sum));

    CHECK(!multimodal::merge_spatial_patches(
        input, 7, grid_height, merge, &merged, &error));
    CHECK(!multimodal::merge_spatial_patches(
        input, grid_width, grid_height, 3, &merged, &error));
    CHECK(!multimodal::run_idefics3_projector(
        input, grid_width, grid_height, merge, projector, nullptr, &error));

    if (failures != 0) return 1;
    std::cout << "multimodal vision projector checks passed\n";
    return 0;
}
