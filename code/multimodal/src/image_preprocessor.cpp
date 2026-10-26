// 多模态 CPU：图像预处理实现
// 职责：实现 half-pixel 双线性 resize、逐通道归一化和 patch 展平。
// 契约：patch 顺序为行优先，每个 patch 内按 C、Y、X 排列。
// 验证：image_preprocessor_test.cpp。
#include "multimodal/image_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace multimodal {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

std::size_t rgb_offset(const RgbImage& image, int x, int y, int channel) {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width) +
            static_cast<std::size_t>(x)) * 3 + static_cast<std::size_t>(channel);
}

}  // namespace

bool ImagePreprocessorConfig::valid(std::string* error) const {
    if (target_width <= 0 || target_height <= 0 || patch_width <= 0 || patch_height <= 0) {
        set_error(error, "target and patch dimensions must be positive");
        return false;
    }
    if (target_width % patch_width != 0 || target_height % patch_height != 0) {
        set_error(error, "target dimensions must be divisible by patch dimensions");
        return false;
    }
    const auto patches_x = static_cast<std::size_t>(target_width / patch_width);
    const auto patches_y = static_cast<std::size_t>(target_height / patch_height);
    const auto patch_area = static_cast<std::size_t>(patch_width) *
                            static_cast<std::size_t>(patch_height);
    if (patches_x > static_cast<std::size_t>(std::numeric_limits<int>::max()) / patches_y ||
        patch_area > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 3) {
        set_error(error, "patch tensor dimensions exceed supported integer range");
        return false;
    }
    for (int channel = 0; channel < 3; ++channel) {
        if (!std::isfinite(mean[channel]) || !std::isfinite(std[channel]) || std[channel] <= 0.0f) {
            set_error(error, "normalization mean must be finite and std must be positive");
            return false;
        }
    }
    return true;
}

bool PatchTensor::valid() const {
    if (patch_count <= 0 || patch_values <= 0) return false;
    const auto count = static_cast<std::size_t>(patch_count);
    const auto width = static_cast<std::size_t>(patch_values);
    return count <= std::numeric_limits<std::size_t>::max() / width &&
           values.size() == count * width;
}

float PatchTensor::at(int patch, int offset) const {
    return values[static_cast<std::size_t>(patch) * static_cast<std::size_t>(patch_values) +
                  static_cast<std::size_t>(offset)];
}

RgbImage resize_bilinear(const RgbImage& input, int target_width, int target_height) {
    if (!input.valid() || target_width <= 0 || target_height <= 0) return {};
    const auto width = static_cast<std::size_t>(target_width);
    const auto height = static_cast<std::size_t>(target_height);
    if (width > std::numeric_limits<std::size_t>::max() / height ||
        width * height > std::numeric_limits<std::size_t>::max() / 3) {
        return {};
    }
    const auto output_size = width * height * 3;
    RgbImage output{target_width, target_height, std::vector<std::uint8_t>(output_size)};

    const float scale_x = static_cast<float>(input.width) / static_cast<float>(target_width);
    const float scale_y = static_cast<float>(input.height) / static_cast<float>(target_height);
    for (int y = 0; y < target_height; ++y) {
        const float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const float clamped_y = std::clamp(source_y, 0.0f, static_cast<float>(input.height - 1));
        const int y0 = static_cast<int>(std::floor(clamped_y));
        const int y1 = std::min(y0 + 1, input.height - 1);
        const float wy = clamped_y - static_cast<float>(y0);
        for (int x = 0; x < target_width; ++x) {
            const float source_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const float clamped_x = std::clamp(source_x, 0.0f, static_cast<float>(input.width - 1));
            const int x0 = static_cast<int>(std::floor(clamped_x));
            const int x1 = std::min(x0 + 1, input.width - 1);
            const float wx = clamped_x - static_cast<float>(x0);
            for (int channel = 0; channel < 3; ++channel) {
                const float top = input.pixels[rgb_offset(input, x0, y0, channel)] * (1.0f - wx) +
                                  input.pixels[rgb_offset(input, x1, y0, channel)] * wx;
                const float bottom = input.pixels[rgb_offset(input, x0, y1, channel)] * (1.0f - wx) +
                                     input.pixels[rgb_offset(input, x1, y1, channel)] * wx;
                const float value = top * (1.0f - wy) + bottom * wy;
                output.pixels[rgb_offset(output, x, y, channel)] =
                    static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 255.0f)));
            }
        }
    }
    return output;
}

FloatImage normalize_chw(const RgbImage& input,
                         const std::array<float, 3>& mean,
                         const std::array<float, 3>& std) {
    if (!input.valid()) return {};
    for (int channel = 0; channel < 3; ++channel) {
        if (!std::isfinite(mean[channel]) || !std::isfinite(std[channel]) || std[channel] <= 0.0f) {
            return {};
        }
    }
    FloatImage output{3, input.width, input.height,
                      std::vector<float>(input.pixels.size())};
    const auto plane = static_cast<std::size_t>(input.width) *
                       static_cast<std::size_t>(input.height);
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            const auto position = static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(input.width) +
                                  static_cast<std::size_t>(x);
            for (int channel = 0; channel < 3; ++channel) {
                const float unit = input.pixels[rgb_offset(input, x, y, channel)] / 255.0f;
                output.values[static_cast<std::size_t>(channel) * plane + position] =
                    (unit - mean[channel]) / std[channel];
            }
        }
    }
    return output;
}

PatchTensor extract_patches(const FloatImage& input, int patch_width, int patch_height) {
    if (!input.valid() || patch_width <= 0 || patch_height <= 0 ||
        input.width % patch_width != 0 || input.height % patch_height != 0) {
        return {};
    }
    const int patches_x = input.width / patch_width;
    const int patches_y = input.height / patch_height;
    PatchTensor output;
    output.patch_count = patches_x * patches_y;
    output.patch_values = input.channels * patch_width * patch_height;
    output.values.resize(static_cast<std::size_t>(output.patch_count) *
                         static_cast<std::size_t>(output.patch_values));

    std::size_t destination = 0;
    for (int patch_y = 0; patch_y < patches_y; ++patch_y) {
        for (int patch_x = 0; patch_x < patches_x; ++patch_x) {
            for (int channel = 0; channel < input.channels; ++channel) {
                for (int y = 0; y < patch_height; ++y) {
                    for (int x = 0; x < patch_width; ++x) {
                        output.values[destination++] = input.at(
                            channel,
                            patch_y * patch_height + y,
                            patch_x * patch_width + x);
                    }
                }
            }
        }
    }
    return output;
}

bool preprocess_image(const RgbImage& input,
                      const ImagePreprocessorConfig& config,
                      PreprocessedImage* output,
                      std::string* error) {
    if (output == nullptr) {
        set_error(error, "preprocessed image output must not be null");
        return false;
    }
    if (!input.valid()) {
        set_error(error, "input RGB image is invalid");
        return false;
    }
    if (!config.valid(error)) return false;

    const auto resized = resize_bilinear(input, config.target_width, config.target_height);
    PreprocessedImage result;
    result.normalized = normalize_chw(resized, config.mean, config.std);
    result.patches = extract_patches(result.normalized, config.patch_width, config.patch_height);
    if (!result.normalized.valid() || !result.patches.valid()) {
        set_error(error, "image preprocessing failed");
        return false;
    }
    *output = std::move(result);
    return true;
}

}  // namespace multimodal
