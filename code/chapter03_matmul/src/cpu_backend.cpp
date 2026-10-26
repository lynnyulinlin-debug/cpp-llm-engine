// Chapter 03：CPU MatMul backend 实现
// 职责：提供 CPU backend 身份，并将公共 MatmulRequest 分派给 reference kernel。
// 契约：不重复实现循环，错误由 kernel 以 Status 返回。
// 验证：chapter03_matmul/scripts/run.sh。
#include "chapter03_matmul/cpu_backend.h"
#include "chapter03_matmul/cpu_kernel.h"

namespace chapter03_matmul {

cpp_llm_engine::core::BackendKind CpuReferenceMatmulBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view CpuReferenceMatmulBackend::name() const {
    return "cpu-reference-matmul";
}

cpp_llm_engine::core::Status CpuReferenceMatmulBackend::compute(
    const cpp_llm_engine::core::MatmulRequest& request) const {
    return cpu_reference_matmul_kernel(request);
}

const CpuReferenceMatmulBackend& cpu_reference_backend() {
    static const CpuReferenceMatmulBackend backend;
    return backend;
}

}  // namespace chapter03_matmul
