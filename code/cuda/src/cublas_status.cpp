// cuBLAS 基础层：实现状态码到稳定错误文本和 core Status 的转换。
// 契约：保留原始 cublasStatus_t，不依赖异常传播库调用失败。
// 验证：cpp_llm_engine_cuda_cublas_matmul_test。
#include "cpp_llm_engine/cuda/cublas_status.h"

#include <utility>

namespace cpp_llm_engine::cuda {
namespace {

const char* cublas_status_name(cublasStatus_t code) {
    switch (code) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "CUBLAS_STATUS_UNKNOWN";
}

}  // namespace

CublasStatus CublasStatus::success() {
    return CublasStatus(CUBLAS_STATUS_SUCCESS, {});
}

CublasStatus CublasStatus::from(cublasStatus_t code, std::string operation) {
    return CublasStatus(code, std::move(operation));
}

CublasStatus::CublasStatus(cublasStatus_t code, std::string operation)
    : code_(code), operation_(std::move(operation)) {}

bool CublasStatus::is_ok() const { return code_ == CUBLAS_STATUS_SUCCESS; }

cublasStatus_t CublasStatus::code() const { return code_; }

const std::string& CublasStatus::operation() const { return operation_; }

std::string CublasStatus::message() const {
    if (is_ok()) return "ok";
    if (operation_.empty()) return cublas_status_name(code_);
    return operation_ + ": " + cublas_status_name(code_);
}

core::Status CublasStatus::to_core_status() const {
    return is_ok() ? core::Status::ok() : core::Status::error(message());
}

}  // namespace cpp_llm_engine::cuda
