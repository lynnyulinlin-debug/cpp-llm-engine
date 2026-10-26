// 多模态 CPU：IDEFICS3 Projector 数学实现
// 职责：实现 llama.cpp pixel-unshuffle 等价的空间合并，再执行 output-major Linear。
// 契约：当前输入网格必须被 merge size 整除，不在数学函数中隐式 padding。
// 验证：multimodal_vision_projector_test 与真实 SmolVLM projector golden。
#include "multimodal/vision_projector.h"

#include <limits>
#include <utility>

namespace multimodal {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

bool checked_product(std::size_t lhs, std::size_t rhs, std::size_t* result) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) return false;
    *result = lhs * rhs;
    return true;
}

}  // namespace

bool merge_spatial_patches(const FloatMatrix& input,
                           std::size_t grid_width,
                           std::size_t grid_height,
                           std::size_t merge_size,
                           FloatMatrix* output,
                           std::string* error) {
    if (output == nullptr) return fail(error, "spatial merge output must not be null");
    std::size_t token_count = 0;
    if (!input.valid() || grid_width == 0 || grid_height == 0 || merge_size == 0 ||
        !checked_product(grid_width, grid_height, &token_count) ||
        input.rows != token_count) {
        return fail(error, "spatial merge input or grid is invalid");
    }
    if (grid_width % merge_size != 0 || grid_height % merge_size != 0) {
        return fail(error, "spatial merge grid must be divisible by merge size");
    }
    std::size_t merged_width = 0;
    if (!checked_product(merge_size, merge_size, &merged_width) ||
        !checked_product(merged_width, input.cols, &merged_width)) {
        return fail(error, "spatial merge output width overflows");
    }
    const std::size_t output_grid_width = grid_width / merge_size;
    const std::size_t output_grid_height = grid_height / merge_size;
    std::size_t output_tokens = 0;
    std::size_t output_elements = 0;
    if (!checked_product(output_grid_width, output_grid_height, &output_tokens) ||
        !checked_product(output_tokens, merged_width, &output_elements)) {
        return fail(error, "spatial merge output shape overflows");
    }

    FloatMatrix merged{output_tokens, merged_width, std::vector<float>(output_elements)};
    for (std::size_t output_y = 0; output_y < output_grid_height; ++output_y) {
        for (std::size_t output_x = 0; output_x < output_grid_width; ++output_x) {
            const std::size_t output_token = output_y * output_grid_width + output_x;
            std::size_t destination = output_token * merged_width;
            for (std::size_t dy = 0; dy < merge_size; ++dy) {
                for (std::size_t dx = 0; dx < merge_size; ++dx) {
                    const std::size_t input_y = output_y * merge_size + dy;
                    const std::size_t input_x = output_x * merge_size + dx;
                    const std::size_t input_token = input_y * grid_width + input_x;
                    for (std::size_t hidden = 0; hidden < input.cols; ++hidden) {
                        merged.values[destination++] = input.at(input_token, hidden);
                    }
                }
            }
        }
    }
    *output = std::move(merged);
    if (error != nullptr) error->clear();
    return true;
}

bool run_idefics3_projector(const FloatMatrix& input,
                            std::size_t grid_width,
                            std::size_t grid_height,
                            std::size_t merge_size,
                            const LinearWeights& projector,
                            Idefics3ProjectorTrace* trace,
                            std::string* error) {
    return run_idefics3_projector_with_backend(
        input, grid_width, grid_height, merge_size, projector,
        VisionComputeBackend::ScalarReference, trace, error);
}

bool run_idefics3_projector_with_backend(const FloatMatrix& input,
                                         std::size_t grid_width,
                                         std::size_t grid_height,
                                         std::size_t merge_size,
                                         const LinearWeights& projector,
                                         VisionComputeBackend backend,
                                         Idefics3ProjectorTrace* trace,
                                         std::string* error) {
    if (trace == nullptr) return fail(error, "IDEFICS3 projector trace must not be null");
    Idefics3ProjectorTrace result;
    if (!merge_spatial_patches(
            input, grid_width, grid_height, merge_size, &result.merged, error) ||
        !apply_linear_with_backend(result.merged, projector, backend, &result.output, error)) {
        return false;
    }
    *trace = std::move(result);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace multimodal
