#pragma once

#include "tensor.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace chapter01 {

class ModelLoader {
public:
    bool load(const std::filesystem::path& path, std::string* error = nullptr);

    const std::unordered_map<std::string, Tensor>& tensors() const;
    const Tensor* find(const std::string& name) const;
    std::vector<std::string> tensor_names() const;

private:
    std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace chapter01

