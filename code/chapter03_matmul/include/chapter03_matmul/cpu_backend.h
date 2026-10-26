// Chapter 03：CPU MatMul backend 接口
// 职责：声明 CPU backend 的设备身份、名称和 MatmulRequest 执行入口。
// 契约：backend 负责适配公共请求，不改变 row-major MatMul 语义。
// 验证：chapter03_matmul_test.cpp 和 core CTest。
#pragma once

#include "cpp_llm_engine/core/matmul_contract.h"

namespace chapter03_matmul {

class CpuReferenceMatmulBackend final : public cpp_llm_engine::core::MatmulBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::MatmulRequest& request) const override;
};

const CpuReferenceMatmulBackend& cpu_reference_backend();

}  // namespace chapter03_matmul
