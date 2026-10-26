// 多模态 CPU：IDEFICS3 动态切图实现
// 职责：实现 token budget 到 refined size 的换算、居中 padding 和 tile crop。
// 契约：overview 位于 tiles 之前；tiles 按 row、column 递增排列。
// 验证：image_preprocessor_test.cpp 与 multimodal_image_inspect。
#include "multimodal/idefics3_preprocessor.h"

#include "multimodal/image_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace multimodal {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

int round_by_factor(double value, int factor) {
    return static_cast<int>(std::round(value / factor)) * factor;
}

int ceil_by_factor(double value, int factor) {
    return static_cast<int>(std::ceil(value / factor)) * factor;
}

int floor_by_factor(double value, int factor) {
    return static_cast<int>(std::floor(value / factor)) * factor;
}

RgbImage resize_with_center_padding(const RgbImage& input,
                                    int target_width,
                                    int target_height,
                                    int* content_width,
                                    int* content_height,
                                    int* offset_x,
                                    int* offset_y) {
    const double scale = std::min(
        static_cast<double>(target_width) / input.width,
        static_cast<double>(target_height) / input.height);
    *content_width = std::min(
        static_cast<int>(std::ceil(input.width * scale)), target_width);
    *content_height = std::min(
        static_cast<int>(std::ceil(input.height * scale)), target_height);
    *offset_x = (target_width - *content_width) / 2;
    *offset_y = (target_height - *content_height) / 2;

    const auto resized = resize_bilinear(input, *content_width, *content_height);
    const auto target_pixels = static_cast<std::size_t>(target_width) *
                               static_cast<std::size_t>(target_height) * 3;
    RgbImage padded{target_width, target_height,
                    std::vector<std::uint8_t>(target_pixels, 0)};
    for (int y = 0; y < resized.height; ++y) {
        const auto source = static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(resized.width) * 3;
        const auto destination =
            (static_cast<std::size_t>(y + *offset_y) * static_cast<std::size_t>(target_width) +
             static_cast<std::size_t>(*offset_x)) * 3;
        std::copy_n(resized.pixels.data() + source,
                    static_cast<std::size_t>(resized.width) * 3,
                    padded.pixels.data() + destination);
    }
    return padded;
}

RgbImage crop_tile(const RgbImage& input, int x, int y, int size) {
    RgbImage tile{size, size,
                  std::vector<std::uint8_t>(static_cast<std::size_t>(size) *
                                            static_cast<std::size_t>(size) * 3)};
    for (int row = 0; row < size; ++row) {
        const auto source =
            (static_cast<std::size_t>(y + row) * static_cast<std::size_t>(input.width) +
             static_cast<std::size_t>(x)) * 3;
        const auto destination = static_cast<std::size_t>(row) *
                                 static_cast<std::size_t>(size) * 3;
        std::copy_n(input.pixels.data() + source,
                    static_cast<std::size_t>(size) * 3,
                    tile.pixels.data() + destination);
    }
    return tile;
}

}  // namespace

bool Idefics3TilingConfig::valid(std::string* error) const {
    if (tile_size <= 0 || patch_size <= 0 || merge_size <= 0 ||
        min_image_tokens <= 0 || max_image_tokens <= 0) {
        set_error(error, "IDEFICS3 tile, patch, merge, and token limits must be positive");
        return false;
    }
    if (tile_size > 8192 || patch_size > tile_size || tile_size % patch_size != 0 ||
        merge_size > tile_size / patch_size || (tile_size / patch_size) % merge_size != 0) {
        set_error(error, "IDEFICS3 tile must be at most 8192 and divisible by patch and merge");
        return false;
    }
    if (min_image_tokens > max_image_tokens) {
        set_error(error, "IDEFICS3 minimum image tokens exceed maximum image tokens");
        return false;
    }
    const auto patch = static_cast<std::int64_t>(patch_size);
    const auto merge = static_cast<std::int64_t>(merge_size);
    const auto patch_area = patch * patch * merge * merge;
    if (patch_area > std::numeric_limits<int>::max() / max_image_tokens) {
        set_error(error, "IDEFICS3 pixel budget exceeds supported integer range");
        return false;
    }
    return true;
}

int Idefics3Layout::visual_image_count() const {
    return 1 + grid_x * grid_y;
}

bool Idefics3Tile::is_overview() const {
    return row < 0 && column < 0;
}

bool calculate_idefics3_layout(int image_width,
                               int image_height,
                               const Idefics3TilingConfig& config,
                               Idefics3Layout* output,
                               std::string* error) {
    if (output == nullptr) {
        set_error(error, "IDEFICS3 layout output must not be null");
        return false;
    }
    if (image_width <= 0 || image_height <= 0) {
        set_error(error, "IDEFICS3 input dimensions must be positive");
        return false;
    }
    if (!config.valid(error)) return false;

    const auto patch_area = config.patch_size * config.patch_size *
                            config.merge_size * config.merge_size;
    const auto min_pixels = config.min_image_tokens * patch_area;
    const auto max_pixels = config.max_image_tokens * patch_area;
    int refined_width = std::max(
        config.tile_size,
        round_by_factor(static_cast<double>(image_width), config.tile_size));
    int refined_height = std::max(
        config.tile_size,
        round_by_factor(static_cast<double>(image_height), config.tile_size));

    const auto aligned_pixels = static_cast<std::int64_t>(refined_width) * refined_height;
    if (aligned_pixels > max_pixels) {
        const double beta = std::sqrt(
            static_cast<double>(image_width) * image_height / max_pixels);
        refined_width = std::max(
            config.tile_size,
            floor_by_factor(image_width / beta, config.tile_size));
        refined_height = std::max(
            config.tile_size,
            floor_by_factor(image_height / beta, config.tile_size));
    } else if (aligned_pixels < min_pixels) {
        const double beta = std::sqrt(
            static_cast<double>(min_pixels) /
            (static_cast<double>(image_width) * image_height));
        refined_width = ceil_by_factor(image_width * beta, config.tile_size);
        refined_height = ceil_by_factor(image_height * beta, config.tile_size);
    }

    Idefics3Layout layout;
    layout.refined_width = refined_width;
    layout.refined_height = refined_height;
    layout.grid_x = refined_width / config.tile_size;
    layout.grid_y = refined_height / config.tile_size;
    const double scale = std::min(
        static_cast<double>(refined_width) / image_width,
        static_cast<double>(refined_height) / image_height);
    layout.content_width = std::min(
        static_cast<int>(std::ceil(image_width * scale)), refined_width);
    layout.content_height = std::min(
        static_cast<int>(std::ceil(image_height * scale)), refined_height);
    layout.offset_x = (refined_width - layout.content_width) / 2;
    layout.offset_y = (refined_height - layout.content_height) / 2;
    *output = layout;
    return true;
}

bool prepare_idefics3_tiles(const RgbImage& input,
                            const Idefics3TilingConfig& config,
                            Idefics3TilingResult* output,
                            std::string* error) {
    if (output == nullptr) {
        set_error(error, "IDEFICS3 tile output must not be null");
        return false;
    }
    if (!input.valid()) {
        set_error(error, "IDEFICS3 input image is invalid");
        return false;
    }

    Idefics3TilingResult result;
    if (!calculate_idefics3_layout(
            input.width, input.height, config, &result.layout, error)) {
        return false;
    }
    result.overview = {-1, -1, resize_bilinear(input, config.tile_size, config.tile_size)};
    const auto refined = resize_with_center_padding(
        input,
        result.layout.refined_width,
        result.layout.refined_height,
        &result.layout.content_width,
        &result.layout.content_height,
        &result.layout.offset_x,
        &result.layout.offset_y);
    result.tiles.reserve(static_cast<std::size_t>(result.layout.grid_x) *
                         static_cast<std::size_t>(result.layout.grid_y));
    for (int row = 0; row < result.layout.grid_y; ++row) {
        for (int column = 0; column < result.layout.grid_x; ++column) {
            result.tiles.push_back({
                row,
                column,
                crop_tile(refined,
                          column * config.tile_size,
                          row * config.tile_size,
                          config.tile_size)});
        }
    }
    *output = std::move(result);
    return true;
}

}  // namespace multimodal
