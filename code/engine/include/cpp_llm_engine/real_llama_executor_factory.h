// Engine：真实 TinyLlama 每请求执行器工厂。
// 职责：共享一份只读 CPU Model，并为每个请求创建独立 Context/Executor。
// 验证：真实服务 CTest 检查双请求 golden token、状态隔离和终态释放。
#pragma once

#include "chapter05_generation/model_executor.h"
#include "cpp_llm_engine/core/decoder_contract.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace cpp_llm_engine {

enum class RealLlamaExecutionMode : std::uint8_t {
    Scalar,
    OptimizedAvx2,
};

class RealLlamaExecutorFactory final {
public:
    static std::unique_ptr<RealLlamaExecutorFactory> load(
        const std::string& model_path,
        std::string* error = nullptr);
    static std::unique_ptr<RealLlamaExecutorFactory> load(
        const std::string& model_path,
        RealLlamaExecutionMode mode,
        std::string* error = nullptr);

    ~RealLlamaExecutorFactory();
    RealLlamaExecutorFactory(RealLlamaExecutorFactory&&) noexcept;
    RealLlamaExecutorFactory& operator=(RealLlamaExecutorFactory&&) noexcept;

    RealLlamaExecutorFactory(const RealLlamaExecutorFactory&) = delete;
    RealLlamaExecutorFactory& operator=(const RealLlamaExecutorFactory&) = delete;

    std::shared_ptr<chapter05_generation::RequestModelExecutor> create_executor() const;
    std::size_t vocab_size() const;
    chapter05_generation::TokenId eos_token_id() const;
    std::size_t layer_count() const;
    cpp_llm_engine::core::DecoderModelContract decoder_contract() const;
    RealLlamaExecutionMode execution_mode() const;

private:
    struct Impl;
    explicit RealLlamaExecutorFactory(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine
