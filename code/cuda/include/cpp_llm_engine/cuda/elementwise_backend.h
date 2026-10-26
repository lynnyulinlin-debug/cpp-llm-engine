// CUDA Backend：声明首个使用公共请求契约的 Vector Add backend。
// 职责：校验 device 指针和 stream，并把非阻塞 kernel 排入指定 stream。
// 验证：CPU golden、非整除 grid、空输入、错误参数和 event 计时。
#pragma once

#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/core/elementwise_contract.h"
#include "cpp_llm_engine/cuda/cuda_status.h"

#include <string_view>

namespace cpp_llm_engine::cuda {

class CudaStream;

class CudaElementwiseBackend final : public core::Backend {
public:
    core::BackendKind kind() const override;
    std::string_view name() const override;

    CudaStatus vector_add(
        const core::VectorAddRequest& request,
        const CudaStream& stream) const;
};

const CudaElementwiseBackend& cuda_elementwise_backend();

}  // namespace cpp_llm_engine::cuda
