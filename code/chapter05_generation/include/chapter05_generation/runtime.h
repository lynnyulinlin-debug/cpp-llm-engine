#pragma once

#include "chapter05_generation/kv_cache.h"
#include "chapter05_generation/generation_types.h"
#include "chapter05_generation/request.h"
#include "chapter05_generation/sampler.h"

namespace chapter05_generation {

class Transformer;

class Runtime {
public:
    explicit Runtime(GenerationConfig config = {});
    Runtime(GenerationConfig config, const Transformer& transformer);

    const GenerationConfig& config() const;
    bool ready() const;
    const KVCache& cache() const;
    KVCache& cache();

    std::size_t prefill(Request& request);
    std::size_t decode_one(Request& request, const Sampler& sampler);
    std::size_t decode_one(Request& request, const Sampler& sampler, const LogitsProvider& logits_provider);
    std::size_t decode(Request& request, const Sampler& sampler);
    std::size_t decode(Request& request, const Sampler& sampler, const LogitsProvider& logits_provider);

    std::size_t generate(
        Request& request,
        const Sampler& sampler,
        const LogitsProvider& logits_provider);

    std::size_t generate(
        Request& request,
        const Sampler& sampler);

private:
    GenerationConfig config_;
    KVCache cache_;
    const Transformer* transformer_ = nullptr;
};

}  // namespace chapter05_generation
