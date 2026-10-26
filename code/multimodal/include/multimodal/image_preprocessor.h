// 多模态 CPU：图像预处理接口
// 职责：完成 RGB resize、CHW normalize 和不重叠 patch 切分。
// 契约：通用预处理不决定 IDEFICS3 等模型特有的动态切图策略。
// 验证：image_preprocessor_test.cpp 与 multimodal_image_inspect。
#pragma once

#include "multimodal/image.h"

#include <array>
#include <string>
#include <vector>

namespace multimodal {

struct ImagePreprocessorConfig {
    int target_width = 0;
    int target_height = 0;
    int patch_width = 0;
    int patch_height = 0;
    std::array<float, 3> mean{0.0f, 0.0f, 0.0f};
    std::array<float, 3> std{1.0f, 1.0f, 1.0f};

    bool valid(std::string* error = nullptr) const;
};

struct PatchTensor {
    int patch_count = 0;
    int patch_values = 0;
    std::vector<float> values;

    bool valid() const;
    float at(int patch, int offset) const;
};

struct PreprocessedImage {
    FloatImage normalized;
    PatchTensor patches;
};

RgbImage resize_bilinear(const RgbImage& input, int target_width, int target_height);
FloatImage normalize_chw(const RgbImage& input,
                         const std::array<float, 3>& mean,
                         const std::array<float, 3>& std);
PatchTensor extract_patches(const FloatImage& input, int patch_width, int patch_height);

bool preprocess_image(const RgbImage& input,
                      const ImagePreprocessorConfig& config,
                      PreprocessedImage* output,
                      std::string* error = nullptr);

}  // namespace multimodal
