// 多模态 CPU：IDEFICS3 Projector 数学接口
// 职责：按空间 4×4 合并视觉 hidden states，并投影到语言模型 embedding width。
// 契约：输入 token 使用 row-major 网格，合并向量内部顺序为 dy、dx、hidden。
// 验证：multimodal_vision_projector_test 与真实 SmolVLM projector golden。
#pragma once

#include "multimodal/vision_encoder.h"

#include <cstddef>
#include <string>

namespace multimodal {

struct Idefics3ProjectorTrace {
    FloatMatrix merged;
    FloatMatrix output;
};

bool merge_spatial_patches(const FloatMatrix& input,
                           std::size_t grid_width,
                           std::size_t grid_height,
                           std::size_t merge_size,
                           FloatMatrix* output,
                           std::string* error = nullptr);

bool run_idefics3_projector(const FloatMatrix& input,
                            std::size_t grid_width,
                            std::size_t grid_height,
                            std::size_t merge_size,
                            const LinearWeights& projector,
                            Idefics3ProjectorTrace* trace,
                            std::string* error = nullptr);

bool run_idefics3_projector_with_backend(const FloatMatrix& input,
                                         std::size_t grid_width,
                                         std::size_t grid_height,
                                         std::size_t merge_size,
                                         const LinearWeights& projector,
                                         VisionComputeBackend backend,
                                         Idefics3ProjectorTrace* trace,
                                         std::string* error = nullptr);

}  // namespace multimodal
