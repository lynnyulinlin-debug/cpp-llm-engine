// CUDA 基础层：保存 CUDA Runtime 错误码和发生操作。
// 职责：让 backend 显式返回设备错误，并在公共边界转换为 core Status。
// 验证：CUDA runtime 测试覆盖成功、人工错误码和 kernel launch 错误。
#pragma once

#include "cpp_llm_engine/core/status.h"

#include <cuda_runtime_api.h>

#include <string>
#include <string_view>

namespace cpp_llm_engine::cuda {

class CudaStatus final {
public:
    static CudaStatus success();
    static CudaStatus from(cudaError_t code, std::string_view operation);

    bool is_ok() const;
    cudaError_t code() const;
    const std::string& operation() const;
    const std::string& message() const;
    core::Status to_core_status() const;

private:
    CudaStatus(cudaError_t code, std::string operation, std::string message);

    cudaError_t code_ = cudaSuccess;
    std::string operation_;
    std::string message_;
};

CudaStatus check_last_cuda_error(std::string_view operation);

}  // namespace cpp_llm_engine::cuda
