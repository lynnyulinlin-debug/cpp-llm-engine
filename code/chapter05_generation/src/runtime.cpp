#include "chapter05_generation/runtime.h"
#include "chapter05_generation/transformer.h"

namespace chapter05_generation {

Runtime::Runtime(GenerationConfig config)
    : config_(config) {}

Runtime::Runtime(GenerationConfig config, const Transformer& transformer)
    : config_(config), transformer_(&transformer) {}

const GenerationConfig& Runtime::config() const {
    return config_;
}

bool Runtime::ready() const {
    return config_.max_new_tokens > 0;
}

const KVCache& Runtime::cache() const {
    return cache_;
}

KVCache& Runtime::cache() {
    return cache_;
}

std::size_t Runtime::prefill(Request& request) {
    if (!ready() || transformer_ == nullptr) {
        return 0;
    }

    const Logits logits = transformer_->prefill(request);
    (void)logits;

    if (config_.use_kv_cache) {
        for (TokenId token : request.prompt()) {
            cache_.append(token);
        }
    }

    request.mark_prefilled();
    return request.prompt_size();
}

std::size_t Runtime::decode_one(Request& request, const Sampler& sampler) {
    if (transformer_ == nullptr) {
        return 0;
    }

    const LogitsProvider logits_provider = [this](const Request& req, std::size_t /*step*/) {
        return transformer_->decode(req);
    };

    return decode_one(request, sampler, logits_provider);
}

std::size_t Runtime::decode_one(
    Request& request,
    const Sampler& sampler,
    const LogitsProvider& logits_provider) {
    if (!ready() || transformer_ == nullptr || !request.prefilled() || request.finished()) {
        return 0;
    }

    request.mark_decoding();
    const std::size_t step = request.generated_size();
    const Logits logits = logits_provider(request, step);
    const TokenId token = sampler.sample(logits);
    request.append_generated_token(token);
    if (config_.use_kv_cache) {
        cache_.append(token);
    }
    if (request.generated_size() >= config_.max_new_tokens) {
        request.mark_finished();
    }
    return 1;
}

std::size_t Runtime::decode(Request& request, const Sampler& sampler) {
    if (transformer_ == nullptr) {
        return 0;
    }

    const LogitsProvider logits_provider = [this](const Request& req, std::size_t step) {
        return transformer_->decode(req);
    };

    return decode(request, sampler, logits_provider);
}

std::size_t Runtime::decode(
    Request& request,
    const Sampler& sampler,
    const LogitsProvider& logits_provider) {
    if (!ready() || transformer_ == nullptr || !request.prefilled()) {
        return 0;
    }

    std::size_t generated_steps = 0;
    while (generated_steps < config_.max_new_tokens && !request.finished()) {
        generated_steps += decode_one(request, sampler, logits_provider);
    }

    return generated_steps;
}

std::size_t Runtime::generate(
    Request& request,
    const Sampler& sampler,
    const LogitsProvider& logits_provider) {
    if (!ready()) {
        return 0;
    }

    std::size_t generated_steps = 0;
    while (generated_steps < config_.max_new_tokens && !request.finished()) {
        const Logits logits = logits_provider(request, generated_steps);
        const TokenId token = sampler.sample(logits);
        request.append_generated_token(token);
        if (config_.use_kv_cache) {
            cache_.append(token);
        }
        ++generated_steps;
    }

    request.mark_finished();
    return generated_steps;
}

std::size_t Runtime::generate(
    Request& request,
    const Sampler& sampler) {
    if (transformer_ == nullptr) {
        return 0;
    }

    prefill(request);
    return decode(request, sampler);
}

}  // namespace chapter05_generation
