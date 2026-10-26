// Engine 测试：验证预计算 embedding 通过既有 Chapter05 Runtime 同步生成。
// 覆盖：shape/position 校验、prefill 后输入释放、decode、EOS 和错误传播。
// 依赖：使用合成 Decoder Context，不依赖真实模型、图片、Python、网络或 GPU。
// 运行：由 CTest 执行 embedding_request_executor_test。
#include "cpp_llm_engine/embedding_request_executor.h"

#include "chapter05_generation/request.h"
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/sampler.h"

#include <cassert>
#include <memory>
#include <string>
#include <utility>

namespace {

class FakeEmbeddingContext final
    : public cpp_llm_engine::core::DecoderExecutionContext {
public:
    void reset() override { token_count_ = 0; }

    cpp_llm_engine::core::Status prefill_tokens(
        const cpp_llm_engine::core::DecoderTokenSequence&,
        cpp_llm_engine::core::DecoderLogits*) override {
        return cpp_llm_engine::core::Status::error("token prefill is not expected");
    }

    cpp_llm_engine::core::Status prefill_embeddings(
        const cpp_llm_engine::core::DecoderEmbeddingSequence& embeddings,
        cpp_llm_engine::core::DecoderLogits* logits) override {
        ++embedding_prefill_calls;
        last_embedding_checksum = 0.0f;
        for (float value : embeddings.values) last_embedding_checksum += value;
        token_count_ = embeddings.token_count;
        logits->assign(5, 0.0f);
        (*logits)[2] = 1.0f;
        return cpp_llm_engine::core::Status::ok();
    }

    cpp_llm_engine::core::Status decode_token(
        cpp_llm_engine::core::DecoderTokenId token,
        cpp_llm_engine::core::DecoderLogits* logits) override {
        ++decode_calls;
        last_decode_token = token;
        ++token_count_;
        logits->assign(5, 0.0f);
        (*logits)[4] = 1.0f;
        return cpp_llm_engine::core::Status::ok();
    }

    std::size_t token_count() const override { return token_count_; }

    cpp_llm_engine::core::DecoderModelContract model_contract() const override {
        cpp_llm_engine::core::DecoderLayerContract layer;
        layer.hidden_size = 4;
        layer.ffn_hidden_size = 8;
        layer.query_head_count = 1;
        layer.kv_head_count = 1;
        layer.head_dimension = 4;
        return {5, 2, 16, layer};
    }

    std::size_t embedding_prefill_calls = 0;
    std::size_t decode_calls = 0;
    cpp_llm_engine::core::DecoderTokenId last_decode_token = -1;
    float last_embedding_checksum = 0.0f;

private:
    std::size_t token_count_ = 0;
};

}  // namespace

int main() {
    cpp_llm_engine::core::DecoderEmbeddingSequence embeddings{
        3, 4, {1.0f, 2.0f, 3.0f, 4.0f,
               5.0f, 6.0f, 7.0f, 8.0f,
               9.0f, 10.0f, 11.0f, 12.0f}};
    auto context = std::make_unique<FakeEmbeddingContext>();
    auto* observed = context.get();
    std::string error;
    auto executor = cpp_llm_engine::EmbeddingRequestExecutor::create(
        std::move(context), std::move(embeddings), &error);
    assert(executor != nullptr && error.empty());
    assert(executor->embedding_storage_bytes() == 12 * sizeof(float));

    chapter05_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 16;
    config.eos_token_id = 4;
    chapter05_generation::Runtime runtime(config);
    chapter05_generation::GreedySampler sampler;
    chapter05_generation::Request request({0, 0, 0});
    assert(runtime.generate(request, sampler, *executor) == 2);
    assert(request.finished());
    assert(request.generated() == chapter05_generation::TokenSequence({2, 4}));
    assert(observed->embedding_prefill_calls == 1);
    assert(observed->decode_calls == 1);
    assert(observed->last_decode_token == 2);
    assert(observed->last_embedding_checksum == 78.0f);
    assert(executor->prefilled());
    assert(executor->embedding_storage_bytes() == 0);
    assert(executor->context_token_count() == 4);

    auto mismatch_context = std::make_unique<FakeEmbeddingContext>();
    auto mismatch = cpp_llm_engine::EmbeddingRequestExecutor::create(
        std::move(mismatch_context), {2, 4, std::vector<float>(8, 1.0f)}, &error);
    chapter05_generation::Logits logits;
    assert(mismatch != nullptr);
    assert(!mismatch->prefill({0}, &logits, &error));
    assert(error.find("position count") != std::string::npos);

    auto invalid_context = std::make_unique<FakeEmbeddingContext>();
    auto invalid = cpp_llm_engine::EmbeddingRequestExecutor::create(
        std::move(invalid_context), {1, 3, std::vector<float>(3, 1.0f)}, &error);
    assert(invalid == nullptr);
    assert(error.find("shape") != std::string::npos);
    return 0;
}
