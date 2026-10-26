// Chapter 01：GGUF tensor payload loader 接口
// 职责：把 GGUF tensor index 映射为教程内部 Tensor 对象。
// 契约：loader 拥有返回 Tensor 的 bytes，不改变 GGUF index。
// 验证：chapter01_tensor_loader_test.cpp。
#pragma once

#include "gguf_reader.h"
#include "tensor.h"

#include <filesystem>
#include <string>
#include <unordered_map>

namespace chapter01 {

class GgufTensorLoader {
public:
    static bool load_tensor(const std::filesystem::path& path,
                            const GgufModelIndex& index,
                            const GgufTensorInfo& info,
                            Tensor* tensor,
                            std::string* error = nullptr);

    static bool load_all(const std::filesystem::path& path,
                         const GgufModelIndex& index,
                         std::unordered_map<std::string, Tensor>* tensors,
                         std::string* error = nullptr);
};

}  // namespace chapter01
