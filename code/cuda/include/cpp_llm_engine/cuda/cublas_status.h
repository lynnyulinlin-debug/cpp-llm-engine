// cuBLAS 基础层：保存原始状态码、操作名和可读错误。
// 职责：让 handle、stream 绑定和 GEMM 失败沿显式状态返回。
// 验证：cuBLAS MatMul CTest 覆盖人工错误、无效 backend 和真实 GEMM。
#pragma once

#include "cpp_llm_engine/core/status.h"

#include <cublas_v2.h>

#include <string>

namespace cpp_llm_engine::cuda {

class CublasStatus final {
public:
    static CublasStatus success();
    static CublasStatus from(cublasStatus_t code, std::string operation);

    bool is_ok() const;
    cublasStatus_t code() const;
    const std::string& operation() const;
    std::string message() const;
    core::Status to_core_status() const;

private:
    CublasStatus(cublasStatus_t code, std::string operation);

    cublasStatus_t code_ = CUBLAS_STATUS_SUCCESS;
    std::string operation_;
};

}  // namespace cpp_llm_engine::cuda
