// 多模态 CPU：视觉 GGUF 权重适配接口
// 职责：加载 F32/Q8_0 payload，并按 SigLIP 数学语义装配 embedding 和单 Block 权重。
// 契约：原始 IDEFICS3 ffn_down 是 MLP 输入投影，ffn_up 是输出投影。
// 验证：multimodal_vision_weights_test 与 multimodal_vision_block_inspect。
#pragma once

#include "gguf_reader.h"
#include "multimodal/mmproj_schema.h"
#include "multimodal/vision_encoder.h"

#include <string>

namespace multimodal {

bool load_patch_embedding_weights(const std::string& path,
                                  const chapter01::GgufModelIndex& index,
                                  const VisionModelConfig& config,
                                  const VisionTensorLayout& layout,
                                  PatchEmbeddingWeights* weights,
                                  std::string* error = nullptr);

bool load_siglip_block_weights(const std::string& path,
                               const chapter01::GgufModelIndex& index,
                               const VisionModelConfig& config,
                               const VisionTensorLayout& layout,
                               std::size_t block_index,
                               SiglipBlockWeights* weights,
                               std::string* error = nullptr);

bool load_post_layer_norm_weights(const std::string& path,
                                  const chapter01::GgufModelIndex& index,
                                  const VisionModelConfig& config,
                                  const VisionTensorLayout& layout,
                                  LayerNormWeights* weights,
                                  std::string* error = nullptr);

bool load_idefics3_projector_weights(const std::string& path,
                                     const chapter01::GgufModelIndex& index,
                                     const VisionModelConfig& config,
                                     const VisionTensorLayout& layout,
                                     LinearWeights* weights,
                                     std::string* error = nullptr);

}  // namespace multimodal
