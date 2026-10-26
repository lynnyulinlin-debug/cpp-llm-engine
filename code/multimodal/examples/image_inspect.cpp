// 多模态 CPU：图像预处理检查工具
// 职责：读取本地图像并打印通用 512x512/16x16 预处理形状和 checksum。
// 运行：multimodal_image_inspect <image.ppm|image.jpg>。
// 验证：输出 RGB、CHW、patch shape 和有限 checksum。
#include "multimodal/image.h"
#include "multimodal/image_preprocessor.h"
#include "multimodal/idefics3_preprocessor.h"

#include <cmath>
#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: multimodal_image_inspect <image.ppm|image.jpg>\n";
        return 2;
    }
    multimodal::RgbImage image;
    std::string error;
    if (!multimodal::load_rgb_image(argv[1], &image, &error)) {
        std::cerr << "decode failed: " << error << '\n';
        return 1;
    }
    const multimodal::ImagePreprocessorConfig config{
        512, 512, 16, 16, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}};
    multimodal::PreprocessedImage processed;
    if (!multimodal::preprocess_image(image, config, &processed, &error)) {
        std::cerr << "preprocess failed: " << error << '\n';
        return 1;
    }
    const multimodal::Idefics3TilingConfig tiling_config{512, 16, 4, 256, 256};
    multimodal::Idefics3TilingResult tiling;
    if (!multimodal::prepare_idefics3_tiles(image, tiling_config, &tiling, &error)) {
        std::cerr << "IDEFICS3 tiling failed: " << error << '\n';
        return 1;
    }
    const double checksum = std::accumulate(
        processed.normalized.values.begin(), processed.normalized.values.end(), 0.0);
    if (!std::isfinite(checksum)) {
        std::cerr << "preprocess produced a non-finite checksum\n";
        return 1;
    }
    std::cout << "rgb=" << image.width << 'x' << image.height << "x3\n"
              << "normalized=3x" << processed.normalized.height << 'x'
              << processed.normalized.width << "\n"
              << "patches=" << processed.patches.patch_count << 'x'
              << processed.patches.patch_values << "\n"
              << "refined=" << tiling.layout.refined_width << 'x'
              << tiling.layout.refined_height << "\n"
              << "content=" << tiling.layout.content_width << 'x'
              << tiling.layout.content_height << '@'
              << tiling.layout.offset_x << ',' << tiling.layout.offset_y << "\n"
              << "grid=" << tiling.layout.grid_x << 'x' << tiling.layout.grid_y << "\n"
              << "visual_images=" << tiling.layout.visual_image_count() << "\n"
              << "checksum=" << checksum << "\n"
              << "jpeg=" << (multimodal::jpeg_decoder_available() ? "enabled" : "disabled")
              << '\n';
    return 0;
}
