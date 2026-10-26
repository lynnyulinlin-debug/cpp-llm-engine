// Chapter 01/04/06：真实 Llama GGUF CPU 模型与请求执行上下文实现。
// 职责：共享量化权重并执行 RMSNorm、RoPE、GQA、SwiGLU 和 LM Head。
// 契约：每个 Context 独占 KV Cache；Model 不随 prefill/decode 改变。
#include "real_llama_cpu_decoder.h"

#include "gguf_reader.h"
#include "gguf_tensor_loader.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"

#include "chapter06_quantization/q4k_dequantize.h"
#include "chapter06_quantization/q4k_linear.h"
#include "chapter06_quantization/q6k_linear.h"
#include "chapter06_quantization/q8_0_dequantize.h"
#include "chapter06_quantization/quantized_linear_policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

namespace chapter01 {
namespace {

using ProfileClock = std::chrono::steady_clock;

class ProfileTimer final {
public:
    explicit ProfileTimer(double* destination)
        : destination_(destination),
          start_(destination == nullptr ? ProfileClock::time_point{} : ProfileClock::now()) {}

    ~ProfileTimer() {
        if (destination_ == nullptr) return;
        *destination_ += std::chrono::duration<double, std::milli>(
            ProfileClock::now() - start_).count();
    }

private:
    double* destination_ = nullptr;
    ProfileClock::time_point start_;
};

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x03ffu;
    std::uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            std::uint32_t normalized = mantissa;
            std::uint32_t exp = 127 - 14;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1;
                --exp;
            }
            result = sign | (exp << 23) | ((normalized & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

const GgufTensorInfo* find_tensor(const GgufModelIndex& index, const std::string& name) {
    for (const auto& tensor : index.tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

bool load_named(const std::string& path,
                const GgufModelIndex& index,
                const std::string& name,
                Tensor* output,
                std::string* error) {
    const auto* info = find_tensor(index, name);
    if (info == nullptr) {
        if (error != nullptr) *error = "missing tensor: " + name;
        return false;
    }
    return GgufTensorLoader::load_tensor(path, index, *info, output, error);
}

std::vector<float> floating_vector(const Tensor& tensor) {
    std::vector<float> result(tensor.numel(), 0.0f);
    if (tensor.dtype() == DType::Float32) {
        std::memcpy(result.data(), tensor.raw_data(), result.size() * sizeof(float));
    } else if (tensor.dtype() == DType::Float16) {
        const auto* values = reinterpret_cast<const std::uint16_t*>(tensor.raw_data());
        for (std::size_t i = 0; i < result.size(); ++i) result[i] = half_to_float(values[i]);
    }
    return result;
}

const chapter06_quantization::QuantizedLinearBackendRegistry&
available_quantized_linear_backends() {
    static const auto registry = [] {
        chapter06_quantization::QuantizedLinearBackendRegistry value;
#if defined(CHAPTER01_HAS_Q4K_AVX2)
        if (chapter06_quantization::q4k_avx2_supported()) {
            value.q4k_avx2 = &chapter06_quantization::q4k_avx2_backend();
        }
#endif
#if defined(CHAPTER01_HAS_Q6K_AVX2)
        if (chapter06_quantization::q6k_avx2_supported()) {
            value.q6k_avx2 = &chapter06_quantization::q6k_avx2_backend();
        }
#endif
        return value;
    }();
    return registry;
}

bool linear(const Tensor& weight,
            const std::vector<float>& input,
            std::vector<float>* output,
            std::string* error,
            const chapter06_quantization::QuantizedLinearPolicy& policy) {
    if (output == nullptr) {
        if (error != nullptr) *error = "linear output must not be null";
        return false;
    }
    const auto& shape = weight.shape();
    if (shape.size() != 2) {
        if (error != nullptr) *error = "expected 2D weight: " + weight.name();
        return false;
    }
    const auto cols = static_cast<std::size_t>(shape[0]);
    const auto rows = static_cast<std::size_t>(shape[1]);
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(weight.raw_data());
    cpp_llm_engine::core::WeightFormat format = cpp_llm_engine::core::WeightFormat::Unknown;
    if (weight.dtype() == DType::Q4_K) format = cpp_llm_engine::core::WeightFormat::Q4K;
    if (weight.dtype() == DType::Q6_K) format = cpp_llm_engine::core::WeightFormat::Q6K;
    if (weight.dtype() == DType::Q8_0) format = cpp_llm_engine::core::WeightFormat::Q8_0;
    if (format == cpp_llm_engine::core::WeightFormat::Unknown) {
        if (error != nullptr) *error = "unsupported linear dtype: " + weight.name();
        return false;
    }
    output->assign(rows, 0.0f);
    const cpp_llm_engine::core::LinearRequest request{
        bytes, weight.bytes(), input.data(), input.size(), output->data(), output->size(),
        rows, cols, format};
    const auto status = chapter06_quantization::compute_quantized_linear(
        request, policy, available_quantized_linear_backends());
    if (!status.is_ok()) {
        output->clear();
        if (error != nullptr) *error = status.message();
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

std::vector<float> rms_norm(const std::vector<float>& input,
                            const std::vector<float>& weight,
                            float epsilon) {
    float mean_square = 0.0f;
    for (float value : input) mean_square += value * value;
    mean_square /= static_cast<float>(input.size());
    const float inverse = 1.0f / std::sqrt(mean_square + epsilon);
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) output[i] = input[i] * inverse * weight[i];
    return output;
}

void apply_rope(std::vector<float>* values,
                std::size_t head_count,
                std::size_t head_dim,
                std::size_t position,
                std::size_t rope_dim,
                float freq_base) {
    const std::size_t applied_dim = std::min(head_dim, rope_dim);
    for (std::size_t head = 0; head < head_count; ++head) {
        float* vector = values->data() + head * head_dim;
        for (std::size_t pair = 0; pair < applied_dim / 2; ++pair) {
            const float exponent = -2.0f * static_cast<float>(pair) /
                                   static_cast<float>(applied_dim);
            const float angle = static_cast<float>(position) * std::pow(freq_base, exponent);
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const float x = vector[2 * pair];
            const float y = vector[2 * pair + 1];
            vector[2 * pair] = x * cosine - y * sine;
            vector[2 * pair + 1] = x * sine + y * cosine;
        }
    }
}

float dot(const float* left, const float* right, std::size_t size) {
    float result = 0.0f;
    for (std::size_t i = 0; i < size; ++i) result += left[i] * right[i];
    return result;
}

void softmax(std::vector<float>* values) {
    const float maximum = *std::max_element(values->begin(), values->end());
    float total = 0.0f;
    for (float& value : *values) {
        value = std::exp(value - maximum);
        total += value;
    }
    for (float& value : *values) value /= total;
}

struct ModelLayer {
    RealLlamaLayerWeights weights;
    std::vector<float> attention_norm;
    std::vector<float> ffn_norm;
};

struct ContextLayer {
    std::vector<float> keys;
    std::vector<float> values;
};

struct ModelData {
    ModelMetadata metadata;
    LlamaTensorLayout layout;
    Tensor embedding;
    Tensor output_weight;
    std::vector<float> output_norm;
    std::vector<ModelLayer> layers;
    std::size_t head_dim = 0;
    std::size_t query_heads = 0;
    std::size_t kv_heads = 0;
    std::size_t rope_dim = 0;
};

bool decode_token_embedding(const ModelData& model,
                            TokenId token,
                            std::vector<float>* output,
                            std::string* error) {
    if (output == nullptr) {
        if (error != nullptr) *error = "token embedding output must not be null";
        return false;
    }
    if (token < 0 || static_cast<std::size_t>(token) >= model.metadata.vocab_size) {
        if (error != nullptr) *error = "token is outside embedding table";
        return false;
    }
    std::size_t block_elements = 0;
    std::size_t block_bytes = 0;
    if (model.embedding.dtype() == DType::Q4_K) {
        block_elements = chapter06_quantization::kQ4KBlockElements;
        block_bytes = chapter06_quantization::kQ4KBlockBytes;
    } else if (model.embedding.dtype() == DType::Q8_0) {
        block_elements = chapter06_quantization::kQ8_0BlockElements;
        block_bytes = chapter06_quantization::kQ8_0BlockBytes;
    } else {
        if (error != nullptr) *error = "embedding table must use Q4_K or Q8_0";
        return false;
    }
    if (model.metadata.hidden_size % block_elements != 0) {
        if (error != nullptr) *error = "embedding width is not quantization-block aligned";
        return false;
    }
    const std::size_t row_bytes = model.metadata.hidden_size / block_elements * block_bytes;
    const std::size_t offset = static_cast<std::size_t>(token) * row_bytes;
    if (offset > model.embedding.bytes() || row_bytes > model.embedding.bytes() - offset) {
        if (error != nullptr) *error = "token embedding payload exceeds table";
        return false;
    }
    const auto* row =
        reinterpret_cast<const std::uint8_t*>(model.embedding.raw_data()) + offset;
    return model.embedding.dtype() == DType::Q4_K
        ? chapter06_quantization::dequantize_q4_k(
              row, row_bytes, model.metadata.hidden_size, output, error)
        : chapter06_quantization::dequantize_q8_0(
              row, row_bytes, model.metadata.hidden_size, output, error);
}

}  // namespace

struct RealLlamaCpuModel::Impl {
    std::shared_ptr<const ModelData> data;
};

struct RealLlamaCpuContext::Impl {
    std::shared_ptr<const ModelData> model;
    std::vector<ContextLayer> layers;
    std::size_t position = 0;
    std::size_t attention_score_count = 0;
    bool profiling_enabled = false;
    chapter06_quantization::QuantizedLinearPolicy attention_linear_policy;
    chapter06_quantization::QuantizedLinearPolicy ffn_linear_policy;
    chapter06_quantization::QuantizedLinearPolicy lm_head_linear_policy;
    RealLlamaDecoderProfile profile;

    bool forward_token(TokenId token, std::vector<float>* logits, std::string* error) {
        ProfileTimer total_timer(profiling_enabled ? &profile.total_ms : nullptr);
        std::vector<float> hidden;
        {
            ProfileTimer timer(profiling_enabled ? &profile.embedding_ms : nullptr);
            if (!decode_token_embedding(*model, token, &hidden, error)) return false;
        }

        return forward_hidden(std::move(hidden), logits, error);
    }

    bool forward_embedding(const float* embedding,
                           std::size_t hidden_size,
                           std::vector<float>* logits,
                           std::string* error) {
        ProfileTimer total_timer(profiling_enabled ? &profile.total_ms : nullptr);
        if (embedding == nullptr || hidden_size != model->metadata.hidden_size) {
            if (error != nullptr) *error = "input embedding width does not match decoder hidden size";
            return false;
        }
        return forward_hidden(
            std::vector<float>(embedding, embedding + hidden_size), logits, error);
    }

    bool forward_hidden(std::vector<float> hidden,
                        std::vector<float>* logits,
                        std::string* error) {
        for (std::size_t layer_index = 0; layer_index < model->layers.size(); ++layer_index) {
            const auto& weights = model->layers[layer_index];
            auto& cache = layers[layer_index];
            std::vector<float> normalized;
            {
                ProfileTimer timer(profiling_enabled ? &profile.normalization_ms : nullptr);
                normalized = rms_norm(hidden, weights.attention_norm,
                                      model->metadata.rms_norm_epsilon);
            }
            std::vector<float> query;
            std::vector<float> key;
            std::vector<float> value;
            {
                ProfileTimer timer(profiling_enabled ? &profile.qkv_projection_ms : nullptr);
                if (!linear(weights.weights.attn_q, normalized, &query, error,
                            attention_linear_policy) ||
                    !linear(weights.weights.attn_k, normalized, &key, error,
                            attention_linear_policy) ||
                    !linear(weights.weights.attn_v, normalized, &value, error,
                            attention_linear_policy)) {
                    return false;
                }
            }
            {
                ProfileTimer timer(profiling_enabled ? &profile.rope_ms : nullptr);
                apply_rope(&query, model->query_heads, model->head_dim, position,
                           model->rope_dim, model->metadata.rope_freq_base);
                apply_rope(&key, model->kv_heads, model->head_dim, position,
                           model->rope_dim, model->metadata.rope_freq_base);
            }
            std::vector<float> attention_output(model->metadata.hidden_size, 0.0f);
            {
                ProfileTimer timer(profiling_enabled ? &profile.attention_ms : nullptr);
                cache.keys.insert(cache.keys.end(), key.begin(), key.end());
                cache.values.insert(cache.values.end(), value.begin(), value.end());
                const float scale = 1.0f / std::sqrt(static_cast<float>(model->head_dim));
                for (std::size_t query_head = 0; query_head < model->query_heads; ++query_head) {
                    const std::size_t kv_head = query_head /
                        (model->query_heads / model->kv_heads);
                    std::vector<float> scores(position + 1, 0.0f);
                    for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                        const float* cached_key = cache.keys.data() +
                            (key_position * model->kv_heads + kv_head) * model->head_dim;
                        scores[key_position] = dot(
                            query.data() + query_head * model->head_dim,
                            cached_key, model->head_dim) * scale;
                    }
                    attention_score_count += scores.size();
                    softmax(&scores);
                    float* output = attention_output.data() + query_head * model->head_dim;
                    for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                        const float* cached_value = cache.values.data() +
                            (key_position * model->kv_heads + kv_head) * model->head_dim;
                        for (std::size_t dimension = 0; dimension < model->head_dim; ++dimension) {
                            output[dimension] += scores[key_position] * cached_value[dimension];
                        }
                    }
                }
            }
            std::vector<float> attention_projection;
            {
                ProfileTimer timer(profiling_enabled ? &profile.attention_output_ms : nullptr);
                if (!linear(weights.weights.attn_output, attention_output,
                            &attention_projection, error, attention_linear_policy)) return false;
                for (std::size_t i = 0; i < hidden.size(); ++i) {
                    hidden[i] += attention_projection[i];
                }
            }
            std::vector<float> ffn_input;
            {
                ProfileTimer timer(profiling_enabled ? &profile.normalization_ms : nullptr);
                ffn_input = rms_norm(hidden, weights.ffn_norm,
                                     model->metadata.rms_norm_epsilon);
            }
            {
                ProfileTimer timer(profiling_enabled ? &profile.ffn_ms : nullptr);
                std::vector<float> gate;
                std::vector<float> up;
                {
                    ProfileTimer sub_timer(
                        profiling_enabled ? &profile.ffn_gate_ms : nullptr);
                    if (!linear(weights.weights.ffn_gate, ffn_input, &gate, error,
                                ffn_linear_policy)) {
                        return false;
                    }
                }
                {
                    ProfileTimer sub_timer(
                        profiling_enabled ? &profile.ffn_up_ms : nullptr);
                    if (!linear(weights.weights.ffn_up, ffn_input, &up, error,
                                ffn_linear_policy)) {
                        return false;
                    }
                }
                {
                    ProfileTimer sub_timer(
                        profiling_enabled ? &profile.ffn_activation_ms : nullptr);
                    for (std::size_t i = 0; i < gate.size(); ++i) {
                        gate[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
                    }
                }
                std::vector<float> ffn_projection;
                {
                    ProfileTimer sub_timer(
                        profiling_enabled ? &profile.ffn_down_ms : nullptr);
                    if (!linear(weights.weights.ffn_down, gate, &ffn_projection, error,
                                ffn_linear_policy)) {
                        return false;
                    }
                }
                {
                    ProfileTimer sub_timer(
                        profiling_enabled ? &profile.ffn_residual_ms : nullptr);
                    for (std::size_t i = 0; i < hidden.size(); ++i) {
                        hidden[i] += ffn_projection[i];
                    }
                }
            }
        }

        ++position;
        if (logits != nullptr) {
            std::vector<float> final_hidden;
            {
                ProfileTimer timer(profiling_enabled ? &profile.normalization_ms : nullptr);
                final_hidden = rms_norm(hidden, model->output_norm,
                                        model->metadata.rms_norm_epsilon);
            }
            {
                ProfileTimer timer(profiling_enabled ? &profile.lm_head_ms : nullptr);
                if (!linear(model->output_weight, final_hidden, logits, error,
                            lm_head_linear_policy)) return false;
            }
            if (logits->size() != model->metadata.vocab_size ||
                !std::all_of(logits->begin(), logits->end(),
                             [](float value) { return std::isfinite(value); })) {
                if (error != nullptr) *error = "decoder produced invalid logits";
                return false;
            }
        }
        if (profiling_enabled) ++profile.forward_tokens;
        return true;
    }
};

std::unique_ptr<RealLlamaCpuModel> RealLlamaCpuModel::load(
    const std::string& model_path,
    std::size_t layer_count,
    std::string* error) {
    GgufModelIndex index;
    if (!GgufReader::read_index(model_path, &index, error)) return nullptr;
    auto data = std::make_shared<ModelData>();
    if (!ModelMetadataAdapter::from_gguf(index.metadata, &data->metadata, error) ||
        !ModelSchema::validate_gguf(data->metadata, index, &data->layout, error)) {
        return nullptr;
    }
    if (layer_count == 0) layer_count = data->metadata.layer_count;
    if (layer_count > data->metadata.layer_count || layer_count == 0) {
        if (error != nullptr) *error = "requested decoder layer count is invalid";
        return nullptr;
    }
    data->query_heads = data->metadata.attention_head_count;
    data->kv_heads = data->metadata.attention_head_count_kv;
    if (data->query_heads == 0 || data->kv_heads == 0 ||
        data->metadata.hidden_size % data->query_heads != 0 ||
        data->query_heads % data->kv_heads != 0) {
        if (error != nullptr) *error = "attention head metadata is invalid";
        return nullptr;
    }
    data->head_dim = data->metadata.hidden_size / data->query_heads;
    data->rope_dim = data->metadata.rope_dimension_count == 0
        ? data->head_dim : data->metadata.rope_dimension_count;
    if (!load_named(model_path, index, data->layout.embedding, &data->embedding, error) ||
        !load_named(model_path, index, data->layout.output, &data->output_weight, error)) {
        return nullptr;
    }
    Tensor output_norm;
    if (!load_named(model_path, index, data->layout.output_norm, &output_norm, error)) {
        return nullptr;
    }
    data->output_norm = floating_vector(output_norm);
    data->layers.resize(layer_count);
    for (std::size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        auto& layer = data->layers[layer_index];
        if (!RealLlamaLayerLoader::load(model_path, index, data->layout, layer_index,
                                        &layer.weights, error)) return nullptr;
        layer.attention_norm = floating_vector(layer.weights.attn_norm);
        layer.ffn_norm = floating_vector(layer.weights.ffn_norm);
    }
    auto impl = std::make_unique<Impl>();
    impl->data = std::move(data);
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealLlamaCpuModel>(new RealLlamaCpuModel(std::move(impl)));
}

RealLlamaCpuModel::RealLlamaCpuModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaCpuModel::~RealLlamaCpuModel() = default;
RealLlamaCpuModel::RealLlamaCpuModel(RealLlamaCpuModel&&) noexcept = default;
RealLlamaCpuModel& RealLlamaCpuModel::operator=(RealLlamaCpuModel&&) noexcept = default;

std::unique_ptr<RealLlamaCpuContext> RealLlamaCpuModel::create_context() const {
    auto context = std::make_unique<RealLlamaCpuContext::Impl>();
    context->model = impl_->data;
    context->layers.resize(impl_->data->layers.size());
    return std::unique_ptr<RealLlamaCpuContext>(
        new RealLlamaCpuContext(std::move(context)));
}

std::size_t RealLlamaCpuModel::layer_count() const { return impl_->data->layers.size(); }
std::size_t RealLlamaCpuModel::model_layer_count() const {
    return impl_->data->metadata.layer_count;
}
std::size_t RealLlamaCpuModel::vocab_size() const { return impl_->data->metadata.vocab_size; }
TokenId RealLlamaCpuModel::eos_token_id() const { return impl_->data->metadata.eos_token_id; }

cpp_llm_engine::core::DecoderModelContract RealLlamaCpuModel::decoder_contract() const {
    return {
        impl_->data->metadata.vocab_size,
        impl_->data->layers.size(),
        impl_->data->metadata.max_position,
        {
            impl_->data->metadata.hidden_size,
            impl_->data->layout.intermediate_size,
            impl_->data->query_heads,
            impl_->data->kv_heads,
            impl_->data->head_dim,
            impl_->data->rope_dim,
            cpp_llm_engine::core::DecoderNormalizationKind::RmsNorm,
            cpp_llm_engine::core::DecoderActivationKind::SwiGlu,
            cpp_llm_engine::core::DecoderPositionEncodingKind::Rotary,
            true,
            true,
        },
    };
}

bool RealLlamaCpuModel::embed_tokens(
    const TokenSequence& tokens,
    cpp_llm_engine::core::DecoderEmbeddingSequence* embeddings,
    std::string* error) const {
    if (tokens.empty() || embeddings == nullptr) {
        if (error != nullptr) *error = "token embedding requires tokens and output";
        return false;
    }
    cpp_llm_engine::core::DecoderEmbeddingSequence result;
    result.token_count = tokens.size();
    result.hidden_size = impl_->data->metadata.hidden_size;
    if (tokens.size() > std::numeric_limits<std::size_t>::max() / result.hidden_size) {
        if (error != nullptr) *error = "token embedding output shape overflows";
        return false;
    }
    result.values.reserve(tokens.size() * result.hidden_size);
    for (const TokenId token : tokens) {
        std::vector<float> row;
        if (!decode_token_embedding(*impl_->data, token, &row, error)) return false;
        result.values.insert(result.values.end(), row.begin(), row.end());
    }
    *embeddings = std::move(result);
    if (error != nullptr) error->clear();
    return true;
}

RealLlamaCpuContext::RealLlamaCpuContext(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaCpuContext::~RealLlamaCpuContext() = default;
RealLlamaCpuContext::RealLlamaCpuContext(RealLlamaCpuContext&&) noexcept = default;
RealLlamaCpuContext& RealLlamaCpuContext::operator=(RealLlamaCpuContext&&) noexcept = default;

void RealLlamaCpuContext::reset() {
    impl_->position = 0;
    impl_->attention_score_count = 0;
    impl_->profile = {};
    for (auto& layer : impl_->layers) {
        layer.keys.clear();
        layer.values.clear();
    }
}

bool RealLlamaCpuContext::prefill(const TokenSequence& tokens,
                                  std::vector<float>* logits,
                                  std::string* error) {
    if (tokens.empty() || logits == nullptr) {
        if (error != nullptr) *error = "prefill requires tokens and logits output";
        return false;
    }
    reset();
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (!impl_->forward_token(tokens[i], i + 1 == tokens.size() ? logits : nullptr,
                                  error)) return false;
    }
    return true;
}

bool RealLlamaCpuContext::decode(TokenId token,
                                 std::vector<float>* logits,
                                 std::string* error) {
    if (impl_->position == 0 || logits == nullptr) {
        if (error != nullptr) *error = "decode requires a completed prefill and logits output";
        return false;
    }
    return impl_->forward_token(token, logits, error);
}

cpp_llm_engine::core::Status RealLlamaCpuContext::prefill_tokens(
    const cpp_llm_engine::core::DecoderTokenSequence& tokens,
    cpp_llm_engine::core::DecoderLogits* logits) {
    return prefill_chunk(tokens, true, true, logits);
}

cpp_llm_engine::core::Status RealLlamaCpuContext::prefill_chunk(
    const cpp_llm_engine::core::DecoderTokenSequence& tokens,
    bool start_new_sequence,
    bool produce_logits,
    cpp_llm_engine::core::DecoderLogits* logits) {
    if (tokens.empty() || (produce_logits && logits == nullptr)) {
        return cpp_llm_engine::core::Status::error(
            "prefill chunk requires tokens and final logits output");
    }
    if (start_new_sequence) {
        reset();
    } else if (impl_->position == 0) {
        return cpp_llm_engine::core::Status::error(
            "appended prefill chunk requires an existing sequence");
    }

    std::string error;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const bool write_logits = produce_logits && i + 1 == tokens.size();
        if (!impl_->forward_token(tokens[i], write_logits ? logits : nullptr, &error)) {
            return cpp_llm_engine::core::Status::error(std::move(error));
        }
    }
    return cpp_llm_engine::core::Status::ok();
}

cpp_llm_engine::core::Status RealLlamaCpuContext::prefill_embeddings(
    const cpp_llm_engine::core::DecoderEmbeddingSequence& embeddings,
    cpp_llm_engine::core::DecoderLogits* logits) {
    const auto validation = embeddings.validate(impl_->model->metadata.hidden_size);
    if (!validation.is_ok()) return validation;
    if (logits == nullptr) {
        return cpp_llm_engine::core::Status::error(
            "embedding prefill requires final logits output");
    }
    reset();
    std::string error;
    for (std::size_t token = 0; token < embeddings.token_count; ++token) {
        const float* row = embeddings.values.data() + token * embeddings.hidden_size;
        const bool write_logits = token + 1 == embeddings.token_count;
        if (!impl_->forward_embedding(
                row, embeddings.hidden_size, write_logits ? logits : nullptr, &error)) {
            return cpp_llm_engine::core::Status::error(std::move(error));
        }
    }
    return cpp_llm_engine::core::Status::ok();
}

cpp_llm_engine::core::Status RealLlamaCpuContext::decode_token(
    cpp_llm_engine::core::DecoderTokenId token,
    cpp_llm_engine::core::DecoderLogits* logits) {
    std::string error;
    if (!decode(token, logits, &error)) {
        return cpp_llm_engine::core::Status::error(std::move(error));
    }
    return cpp_llm_engine::core::Status::ok();
}

std::size_t RealLlamaCpuContext::token_count() const {
    return impl_->position;
}

cpp_llm_engine::core::DecoderModelContract RealLlamaCpuContext::model_contract() const {
    return {
        impl_->model->metadata.vocab_size,
        impl_->model->layers.size(),
        impl_->model->metadata.max_position,
        {
            impl_->model->metadata.hidden_size,
            impl_->model->layout.intermediate_size,
            impl_->model->query_heads,
            impl_->model->kv_heads,
            impl_->model->head_dim,
            impl_->model->rope_dim,
            cpp_llm_engine::core::DecoderNormalizationKind::RmsNorm,
            cpp_llm_engine::core::DecoderActivationKind::SwiGlu,
            cpp_llm_engine::core::DecoderPositionEncodingKind::Rotary,
            true,
            true,
        },
    };
}

void RealLlamaCpuContext::set_profiling_enabled(bool enabled) {
    impl_->profiling_enabled = enabled;
    impl_->profile = {};
}

bool RealLlamaCpuContext::set_attention_linear_policy(
    const chapter06_quantization::QuantizedLinearPolicy& policy,
    std::string* error) {
    const auto status = chapter06_quantization::validate_quantized_linear_policy(
        policy, available_quantized_linear_backends());
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return false;
    }
    impl_->attention_linear_policy = policy;
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaCpuContext::set_ffn_linear_policy(
    const chapter06_quantization::QuantizedLinearPolicy& policy,
    std::string* error) {
    const auto status = chapter06_quantization::validate_quantized_linear_policy(
        policy, available_quantized_linear_backends());
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return false;
    }
    impl_->ffn_linear_policy = policy;
    if (error != nullptr) error->clear();
    return true;
}

RealLlamaDecoderStats RealLlamaCpuContext::stats() const {
    return RealLlamaDecoderStats{
        impl_->model->metadata.layer_count,
        impl_->model->layers.size(),
        impl_->position,
        impl_->model->query_heads,
        impl_->model->kv_heads,
        impl_->model->head_dim,
        impl_->model->rope_dim,
        impl_->attention_score_count,
        impl_->model->metadata.rope_freq_base,
    };
}

RealLlamaDecoderProfile RealLlamaCpuContext::profile() const {
    return impl_->profile;
}

}  // namespace chapter01
