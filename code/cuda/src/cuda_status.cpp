// CUDA 基础层：实现 CUDA Runtime 错误到稳定文本和 core Status 的转换。
// 契约：成功没有错误文本；失败保留原始错误码、操作名和 runtime 描述。
// 验证：cpp_llm_engine_cuda_test 检查人工错误与真实 launch 错误。
#include "cpp_llm_engine/cuda/cuda_status.h"

#include <utility>

namespace cpp_llm_engine::cuda {

CudaStatus::CudaStatus(cudaError_t code, std::string operation, std::string message)
    : code_(code), operation_(std::move(operation)), message_(std::move(message)) {}

CudaStatus CudaStatus::success() {
    return CudaStatus(cudaSuccess, {}, {});
}

CudaStatus CudaStatus::from(cudaError_t code, std::string_view operation) {
    if (code == cudaSuccess) return success();
    std::string message(operation);
    if (!message.empty()) message += ": ";
    message += cudaGetErrorString(code);
    return CudaStatus(code, std::string(operation), std::move(message));
}

bool CudaStatus::is_ok() const { return code_ == cudaSuccess; }

cudaError_t CudaStatus::code() const { return code_; }

const std::string& CudaStatus::operation() const { return operation_; }

const std::string& CudaStatus::message() const { return message_; }

core::Status CudaStatus::to_core_status() const {
    return is_ok() ? core::Status::ok() : core::Status::error(message_);
}

CudaStatus check_last_cuda_error(std::string_view operation) {
    return CudaStatus::from(cudaGetLastError(), operation);
}

}  // namespace cpp_llm_engine::cuda
