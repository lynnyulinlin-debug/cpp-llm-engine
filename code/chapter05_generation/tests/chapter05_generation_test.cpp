// Chapter 05：测试从 prefill 到 decode、completion 的 Generation Core path。
// 覆盖：output length、cache growth、terminal state 和 context admission。
// 验证入口：Chapter 05 CTest target 或 scripts/run.sh。
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/transformer.h"
#include "chapter05_generation/sampler.h"

#include <cassert>

int main() {
    chapter05_generation::GenerationConfig config;
    config.max_new_tokens = 3;
    config.use_kv_cache = true;

    chapter05_generation::Request request({1, 2, 3});
    chapter05_generation::GreedySampler sampler;
    chapter05_generation::Embedding embedding(32);
    chapter05_generation::PositionEncoding position_encoding(16);
    chapter05_generation::Transformer transformer(2, embedding, position_encoding);
    chapter05_generation::Runtime runtime(config, transformer);

    assert(embedding.ready());
    assert(position_encoding.ready());
    assert(transformer.ready());
    assert(transformer.layer_count() == 2);
    assert(transformer.embedding().vocab_size() == 32);
    assert(transformer.position_encoding().max_length() == 16);

    const std::size_t prefills = runtime.prefill(request);
    assert(prefills == 3);
    assert(request.prefilled());
    assert(request.phase() == chapter05_generation::RequestPhase::Prefilled);
    assert(request.cache().size() == 3);
    assert(runtime.prefill(request) == 0);
    assert(request.cache().size() == 3);

    const std::size_t generated_steps = runtime.decode(request, sampler);
    assert(generated_steps == 3);
    assert(request.generated_size() == 3);
    assert(request.phase() == chapter05_generation::RequestPhase::Finished);
    assert(request.cache().size() == 6);
    assert(request.finished());

    chapter05_generation::GenerationConfig limited_config;
    limited_config.max_new_tokens = 2;
    limited_config.max_context_tokens = 4;
    chapter05_generation::Runtime limited_runtime(limited_config, transformer);
    chapter05_generation::Request rejected({1, 2, 3});
    assert(limited_runtime.prefill(rejected) == 0);
    assert(rejected.phase() == chapter05_generation::RequestPhase::Rejected);
    assert(rejected.terminal());
    assert(rejected.cache().empty());

    chapter05_generation::GenerationConfig provider_config;
    provider_config.max_new_tokens = 4;
    provider_config.eos_token_id = 7;
    chapter05_generation::Runtime provider_runtime(provider_config);
    chapter05_generation::Request provider_request({4, 5});
    const auto provider = [](const chapter05_generation::Request&, std::size_t) {
        chapter05_generation::Logits logits(8, 0.0f);
        logits[7] = 1.0f;
        return logits;
    };
    assert(provider_runtime.generate(provider_request, sampler, provider) == 1);
    assert(provider_request.generated() == chapter05_generation::TokenSequence({7}));
    assert(provider_request.cache().size() == 3);
    assert(provider_request.finished());
    return 0;
}
