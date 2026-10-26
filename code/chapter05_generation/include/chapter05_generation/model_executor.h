// Chapter 05：声明单请求模型执行器契约。
// 职责：为一个 Request 持有模型侧状态，并向 Runtime 提供 prefill/decode logits。
// 生命周期：ServiceCore 或调用方持有 executor，terminal cleanup 时与请求一起释放。
#pragma once

#include "chapter05_generation/generation_types.h"

#include <string>

namespace chapter05_generation {

class RequestModelExecutor {
public:
    virtual ~RequestModelExecutor() = default;

    virtual bool prefill(const TokenSequence& prompt,
                         Logits* logits,
                         std::string* error) = 0;
    virtual bool supports_chunked_prefill() const { return false; }
    virtual bool prefill_chunk(const TokenSequence& prompt,
                               bool start_new_sequence,
                               bool produce_logits,
                               Logits* logits,
                               std::string* error) {
        if (!start_new_sequence || !produce_logits) {
            if (error != nullptr) {
                *error = "model executor does not support chunked prefill";
            }
            return false;
        }
        return prefill(prompt, logits, error);
    }
    virtual bool decode(TokenId token,
                        Logits* logits,
                        std::string* error) = 0;
};

}  // namespace chapter05_generation
