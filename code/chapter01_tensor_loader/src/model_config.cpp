// Chapter 01：模型配置实现
// 职责：校验教学模型的 vocab、hidden、layer、context 和 generation 配置。
// 契约：无效配置返回明确错误，供 Engine 和 Runtime 在构造前检查。
// 验证：chapter01_tensor_loader_test.cpp 与 Engine smoke test。
#include "model_config.h"

#include <sstream>

namespace chapter01 {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
    return false;
}

}  // namespace

ModelConfig ModelConfig::from_metadata(const ModelMetadata& metadata) {
    ModelConfig config;
    config.architecture = metadata.architecture;
    config.vocab_size = metadata.vocab_size;
    config.hidden_size = metadata.hidden_size;
    config.max_position = metadata.max_position;
    config.layer_count = metadata.layer_count;
    config.rms_norm_epsilon = metadata.rms_norm_epsilon;
    config.bos_token_id = metadata.bos_token_id;
    config.eos_token_id = metadata.eos_token_id;
    return config;
}

bool ModelConfig::valid() const {
    return !architecture.empty() && vocab_size > 1 && hidden_size > 0 && max_position > 0 &&
           layer_count > 0 && max_new_tokens > 0;
}

bool ModelConfig::validate_tokenizer(const Tokenizer& tokenizer, std::string* error) const {
    if (!valid()) {
        return fail(error, "invalid model configuration");
    }
    if (tokenizer.vocab_size() != vocab_size) {
        std::ostringstream message;
        message << "tokenizer vocab size " << tokenizer.vocab_size()
                << " does not match model vocab size " << vocab_size;
        return fail(error, message.str());
    }
    return true;
}

bool ModelConfig::validate_tensor(const Tensor& tensor, std::string* error) const {
    if (!valid()) {
        return fail(error, "invalid model configuration");
    }
    if (tensor.name().empty() || tensor.shape().empty() || tensor.dtype() == DType::Unknown) {
        return fail(error, "tensor metadata is incomplete");
    }
    if (tensor.name() == "embedding.weight" || tensor.name() == "lm_head.weight") {
        const auto& shape = tensor.shape();
        if (shape.size() != 2 || shape[0] != static_cast<std::int64_t>(vocab_size) ||
            shape[1] != static_cast<std::int64_t>(hidden_size)) {
            return fail(error, "embedding or lm_head shape does not match model configuration");
        }
    }
    return true;
}

}  // namespace chapter01
