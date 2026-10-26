// 多模态 CPU：IDEFICS3 动态切图接口
// 职责：按图像 token 预算计算 refined image、overview 和行优先 tile。
// 契约：只负责几何与 RGB tile，不执行 Vision Encoder 或 visual token 拼接。
// 验证：image_preprocessor_test.cpp 与真实 640x488 JPEG inspect。
#pragma once

#include "multimodal/image.h"

#include <string>
#include <vector>

namespace multimodal {

struct Idefics3TilingConfig {
    int tile_size = 0;
    int patch_size = 0;
    int merge_size = 1;
    int min_image_tokens = 0;
    int max_image_tokens = 0;

    bool valid(std::string* error = nullptr) const;
};

struct Idefics3Layout {
    int refined_width = 0;
    int refined_height = 0;
    int content_width = 0;
    int content_height = 0;
    int offset_x = 0;
    int offset_y = 0;
    int grid_x = 0;
    int grid_y = 0;

    int visual_image_count() const;
};

struct Idefics3Tile {
    int row = -1;
    int column = -1;
    RgbImage image;

    bool is_overview() const;
};

struct Idefics3TilingResult {
    Idefics3Layout layout;
    Idefics3Tile overview;
    std::vector<Idefics3Tile> tiles;
};

bool calculate_idefics3_layout(int image_width,
                               int image_height,
                               const Idefics3TilingConfig& config,
                               Idefics3Layout* output,
                               std::string* error = nullptr);

bool prepare_idefics3_tiles(const RgbImage& input,
                            const Idefics3TilingConfig& config,
                            Idefics3TilingResult* output,
                            std::string* error = nullptr);

}  // namespace multimodal

