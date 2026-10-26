// Chapter 05：实现单个 Request 的 CPU prefill、decode 和 generation orchestration。
// 职责：在委托 model 与 sampling work 前执行 admission 和 terminal-state 规则。
// 验证：Runtime smoke、generation 和 streaming 测试检查可观察生命周期。
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/transformer.h"

#include <algorithm>
#include <utility>

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

std::size_t Runtime::prefill(Request& request) {
    if (!ready() || transformer_ == nullptr || request.terminal() || request.prefilled()) {
        return 0;
    }
    if (config_.max_context_tokens > 0 &&
        (request.prompt_size() > config_.max_context_tokens ||
         config_.max_new_tokens > config_.max_context_tokens - request.prompt_size())) {
        request.reject();
        return 0;
    }

    const Logits logits = transformer_->prefill(request);
    (void)logits;

    if (config_.use_kv_cache) {
        for (TokenId token : request.prompt()) {
            request.cache().append(token);
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
    if (!ready() || transformer_ == nullptr || !request.prefilled() || request.terminal()) {
        return 0;
    }

    request.mark_decoding();
    const std::size_t step = request.generated_size();
    const Logits logits = logits_provider(request, step);
    const TokenId token = sampler.sample(logits);
    request.append_generated_token(token);
    if (config_.use_kv_cache) {
        request.cache().append(token);
    }
    if (token == config_.eos_token_id ||
        request.generated_size() >= config_.max_new_tokens) {
        request.mark_finished();
    }
    return 1;
}

std::size_t Runtime::decode_one(
    Request& request,
    const Sampler& sampler,
    RequestModelExecutor& executor) {
    if (!ready() || request.terminal()) return 0;
    if (!request.prefilled() && config_.max_context_tokens > 0 &&
        (request.prompt_size() > config_.max_context_tokens ||
         config_.max_new_tokens > config_.max_context_tokens - request.prompt_size())) {
        request.reject();
        return 0;
    }

    if (!request.prefilled()) {
        return prefill_chunk(
            request, sampler, executor, request.remaining_prompt_tokens()).generated_tokens;
    }

    if (request.generated().empty()) {
        request.mark_failed("model decode requires a previously generated token");
        return 0;
    }

    Logits logits;
    std::string error;
    if (!executor.decode(request.generated().back(), &logits, &error)) {
        request.mark_failed(error.empty() ? "model decode failed" : std::move(error));
        return 0;
    }
    if (logits.empty()) {
        request.mark_failed("model executor returned empty logits");
        return 0;
    }

    request.mark_decoding();
    const TokenId token = sampler.sample(logits);
    request.append_generated_token(token);
    if (config_.use_kv_cache) request.cache().append(token);
    if (token == config_.eos_token_id ||
        request.generated_size() >= config_.max_new_tokens) {
        request.mark_finished();
    }
    return 1;
}

PrefillChunkResult Runtime::prefill_chunk(
    Request& request,
    const Sampler& sampler,
    RequestModelExecutor& executor,
    std::size_t max_prompt_tokens) {
    PrefillChunkResult result;
    if (!ready() || request.terminal() || request.prefilled() || max_prompt_tokens == 0) {
        return result;
    }
    if (config_.max_context_tokens > 0 &&
        (request.prompt_size() > config_.max_context_tokens ||
         config_.max_new_tokens > config_.max_context_tokens - request.prompt_size())) {
        request.reject();
        return result;
    }

    const std::size_t remaining = request.remaining_prompt_tokens();
    const std::size_t chunk_size = std::min(remaining, max_prompt_tokens);
    if (chunk_size == 0) {
        request.mark_failed("model prefill requires a non-empty prompt");
        return result;
    }
    const bool starts_sequence = request.prefill_offset() == 0;
    const bool completes_prefill = chunk_size == remaining;
    if (!completes_prefill && !executor.supports_chunked_prefill()) {
        return result;
    }

    const auto first = request.prompt().begin() +
        static_cast<TokenSequence::difference_type>(request.prefill_offset());
    const TokenSequence chunk(first, first +
        static_cast<TokenSequence::difference_type>(chunk_size));
    Logits logits;
    std::string error;
    if (!executor.prefill_chunk(
            chunk,
            starts_sequence,
            completes_prefill,
            completes_prefill ? &logits : nullptr,
            &error)) {
        request.mark_failed(error.empty() ? "model prefill failed" : std::move(error));
        return result;
    }
    if (completes_prefill && logits.empty()) {
        request.mark_failed("model executor returned empty logits");
        return result;
    }

    request.advance_prefill(chunk_size);
    result.consumed_tokens = chunk_size;
    if (config_.use_kv_cache) {
        for (TokenId token : chunk) request.cache().append(token);
    }
    if (!completes_prefill) return result;

    request.mark_prefilled();
    request.mark_decoding();
    const TokenId token = sampler.sample(logits);
    request.append_generated_token(token);
    if (config_.use_kv_cache) request.cache().append(token);
    if (token == config_.eos_token_id ||
        request.generated_size() >= config_.max_new_tokens) {
        request.mark_finished();
    }
    result.generated_tokens = 1;
    return result;
}

std::size_t Runtime::decode(Request& request, const Sampler& sampler) {
    if (transformer_ == nullptr) {
        return 0;
    }

    const LogitsProvider logits_provider = [this](const Request& req, std::size_t /*step*/) {
        return transformer_->decode(req);
    };

    return decode(request, sampler, logits_provider);
}

std::size_t Runtime::decode(
    Request& request,
    const Sampler& sampler,
    const LogitsProvider& logits_provider) {
    if (!ready() || transformer_ == nullptr || !request.prefilled() || request.terminal()) {
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
    if (!ready() || request.terminal()) {
        return 0;
    }
    if (config_.max_context_tokens > 0 &&
        (request.prompt_size() > config_.max_context_tokens ||
         config_.max_new_tokens > config_.max_context_tokens - request.prompt_size())) {
        request.reject();
        return 0;
    }
    if (!request.prefilled()) {
        if (config_.use_kv_cache) {
            for (TokenId token : request.prompt()) request.cache().append(token);
        }
        request.mark_prefilled();
    }

    std::size_t generated_steps = 0;
    while (generated_steps < config_.max_new_tokens && !request.finished()) {
        request.mark_decoding();
        const Logits logits = logits_provider(request, generated_steps);
        const TokenId token = sampler.sample(logits);
        request.append_generated_token(token);
        if (config_.use_kv_cache) {
            request.cache().append(token);
        }
        ++generated_steps;
        if (token == config_.eos_token_id) request.mark_finished();
    }

    if (!request.finished()) request.mark_finished();
    return generated_steps;
}

std::size_t Runtime::generate(
    Request& request,
    const Sampler& sampler,
    RequestModelExecutor& executor) {
    std::size_t generated_steps = 0;
    while (generated_steps < config_.max_new_tokens && !request.terminal()) {
        const std::size_t decoded = decode_one(request, sampler, executor);
        if (decoded == 0) break;
        generated_steps += decoded;
    }
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
