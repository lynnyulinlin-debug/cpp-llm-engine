// Chapter 01：GGUF metadata 适配接口
// 职责：声明 GGUF metadata 到 ModelMetadata 的转换边界。
// 契约：适配器不读取 tensor payload，只消费已解析 metadata。
// 验证：chapter01_tensor_loader_test.cpp。
#pragma once

#include "model_metadata.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace chapter01 {

using MetadataValue = std::variant<std::string,
                                   std::int64_t,
                                   std::uint64_t,
                                   float,
                                   std::vector<std::string>,
                                   std::vector<std::int32_t>,
                                   std::vector<std::uint32_t>,
                                   std::vector<float>>;
using MetadataMap = std::unordered_map<std::string, MetadataValue>;

// Converts parsed GGUF key/value metadata into the format-independent model contract.
class ModelMetadataAdapter {
public:
    static bool from_gguf(const MetadataMap& values,
                          ModelMetadata* metadata,
                          std::string* error = nullptr);
};

}  // namespace chapter01
