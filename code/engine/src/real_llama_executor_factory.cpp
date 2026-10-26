// Engine：实现真实 TinyLlama Model 到 RequestModelExecutor 的适配。
// 所有权：Factory 持有 Model；每个 Executor 持有独立 Context 及其分层 KV。
// 验证：真实生成和真实服务 CTest 共同覆盖 prefill/decode 与请求隔离。
#include "cpp_llm_engine/real_llama_executor_factory.h"

#include "real_llama_cpu_decoder.h"

#include <utility>

namespace cpp_llm_engine {
namespace {

bool configure_context(
    chapter01::RealLlamaCpuContext* context,
    RealLlamaExecutionMode mode,
    std::string* error) {
    const auto scalar = chapter06_quantization::QuantizedLinearBackendKind::ScalarFused;
    const auto q4_backend = mode == RealLlamaExecutionMode::OptimizedAvx2
        ? chapter06_quantization::QuantizedLinearBackendKind::Avx2Fused : scalar;
    const auto q6_backend = mode == RealLlamaExecutionMode::OptimizedAvx2
        ? chapter06_quantization::QuantizedLinearBackendKind::Avx2Fused : scalar;
    const chapter06_quantization::QuantizedLinearPolicy attention_policy{
        q4_backend, scalar};
    const chapter06_quantization::QuantizedLinearPolicy ffn_policy{
        q4_backend, q6_backend};
    return context->set_attention_linear_policy(attention_policy, error) &&
        context->set_ffn_linear_policy(ffn_policy, error);
}

class RealLlamaRequestExecutor final
    : public chapter05_generation::RequestModelExecutor {
public:
    explicit RealLlamaRequestExecutor(
        std::unique_ptr<cpp_llm_engine::core::DecoderExecutionContext> context)
        : context_(std::move(context)) {}

    bool prefill(const chapter05_generation::TokenSequence& prompt,
                 chapter05_generation::Logits* logits,
                 std::string* error) override {
        const auto status = context_->prefill_tokens(prompt, logits);
        if (!status.is_ok() && error != nullptr) *error = status.message();
        return status.is_ok();
    }

    bool supports_chunked_prefill() const override { return true; }

    bool prefill_chunk(const chapter05_generation::TokenSequence& prompt,
                       bool start_new_sequence,
                       bool produce_logits,
                       chapter05_generation::Logits* logits,
                       std::string* error) override {
        const auto status = context_->prefill_chunk(
            prompt, start_new_sequence, produce_logits, logits);
        if (!status.is_ok() && error != nullptr) *error = status.message();
        return status.is_ok();
    }

    bool decode(chapter05_generation::TokenId token,
                chapter05_generation::Logits* logits,
                std::string* error) override {
        const auto status = context_->decode_token(token, logits);
        if (!status.is_ok() && error != nullptr) *error = status.message();
        return status.is_ok();
    }

private:
    std::unique_ptr<cpp_llm_engine::core::DecoderExecutionContext> context_;
};

}  // namespace

struct RealLlamaExecutorFactory::Impl {
    std::unique_ptr<chapter01::RealLlamaCpuModel> model;
    RealLlamaExecutionMode mode = RealLlamaExecutionMode::Scalar;
};

std::unique_ptr<RealLlamaExecutorFactory> RealLlamaExecutorFactory::load(
    const std::string& model_path,
    std::string* error) {
    return load(model_path, RealLlamaExecutionMode::Scalar, error);
}

std::unique_ptr<RealLlamaExecutorFactory> RealLlamaExecutorFactory::load(
    const std::string& model_path,
    RealLlamaExecutionMode mode,
    std::string* error) {
    auto model = chapter01::RealLlamaCpuModel::load(model_path, 0, error);
    if (model == nullptr) return nullptr;
    auto validation_context = model->create_context();
    if (!configure_context(validation_context.get(), mode, error)) return nullptr;
    auto impl = std::make_unique<Impl>();
    impl->model = std::move(model);
    impl->mode = mode;
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealLlamaExecutorFactory>(
        new RealLlamaExecutorFactory(std::move(impl)));
}

RealLlamaExecutorFactory::RealLlamaExecutorFactory(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaExecutorFactory::~RealLlamaExecutorFactory() = default;
RealLlamaExecutorFactory::RealLlamaExecutorFactory(
    RealLlamaExecutorFactory&&) noexcept = default;
RealLlamaExecutorFactory& RealLlamaExecutorFactory::operator=(
    RealLlamaExecutorFactory&&) noexcept = default;

std::shared_ptr<chapter05_generation::RequestModelExecutor>
RealLlamaExecutorFactory::create_executor() const {
    auto context = impl_->model->create_context();
    std::string error;
    if (!configure_context(context.get(), impl_->mode, &error)) return nullptr;
    return std::make_shared<RealLlamaRequestExecutor>(std::move(context));
}

std::size_t RealLlamaExecutorFactory::vocab_size() const {
    return impl_->model->vocab_size();
}

chapter05_generation::TokenId RealLlamaExecutorFactory::eos_token_id() const {
    return impl_->model->eos_token_id();
}

std::size_t RealLlamaExecutorFactory::layer_count() const {
    return impl_->model->layer_count();
}

cpp_llm_engine::core::DecoderModelContract
RealLlamaExecutorFactory::decoder_contract() const {
    return impl_->model->decoder_contract();
}

RealLlamaExecutionMode RealLlamaExecutorFactory::execution_mode() const {
    return impl_->mode;
}

}  // namespace cpp_llm_engine
