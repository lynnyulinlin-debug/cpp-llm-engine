// Chapter 05：声明负责 prefill、decode 和 token generation 的 CPU Generation Runtime。
// 职责：推进 Request state、写入 request cache，并把 token choice 交给 Sampler。
// 验证：Runtime smoke 和 generation 测试检查 admission、terminal handling 和 output count。
#pragma once

#include "chapter05_generation/generation_types.h"
#include "chapter05_generation/model_executor.h"
#include "chapter05_generation/request.h"
#include "chapter05_generation/sampler.h"

namespace chapter05_generation {

class Transformer;

struct PrefillChunkResult {
    std::size_t consumed_tokens = 0;
    std::size_t generated_tokens = 0;
};

class Runtime {
public:
    explicit Runtime(GenerationConfig config = {});
    Runtime(GenerationConfig config, const Transformer& transformer);

    const GenerationConfig& config() const;
    bool ready() const;
    std::size_t prefill(Request& request);
    std::size_t decode_one(Request& request, const Sampler& sampler);
    std::size_t decode_one(Request& request, const Sampler& sampler, const LogitsProvider& logits_provider);
    std::size_t decode_one(Request& request, const Sampler& sampler, RequestModelExecutor& executor);
    PrefillChunkResult prefill_chunk(
        Request& request,
        const Sampler& sampler,
        RequestModelExecutor& executor,
        std::size_t max_prompt_tokens);
    std::size_t decode(Request& request, const Sampler& sampler);
    std::size_t decode(Request& request, const Sampler& sampler, const LogitsProvider& logits_provider);

    std::size_t generate(
        Request& request,
        const Sampler& sampler,
        const LogitsProvider& logits_provider);
    std::size_t generate(
        Request& request,
        const Sampler& sampler,
        RequestModelExecutor& executor);

    std::size_t generate(
        Request& request,
        const Sampler& sampler);

private:
    GenerationConfig config_;
    const Transformer* transformer_ = nullptr;
};

}  // namespace chapter05_generation
