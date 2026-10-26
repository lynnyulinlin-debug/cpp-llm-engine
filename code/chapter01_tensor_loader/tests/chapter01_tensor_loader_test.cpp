// Chapter 01：模型加载与 schema 综合测试
// 职责：验证 CIE1 loader、GGUF index、metadata、tokenizer、Q8_0/Q4_K 和 DecoderInput。
// 通过条件：fixture 与非法输入断言全部成立。
// 验证：bash code/chapter01_tensor_loader/run.sh。
#include "../src/model_loader.h"
#include "../src/model_config.h"
#include "../src/model_metadata.h"
#include "../src/model_metadata_adapter.h"
#include "../src/gguf_reader.h"
#include "../src/gguf_tensor_loader.h"
#include "../src/model_builder.h"
#include "../src/tokenizer_metadata.h"
#include "../src/tensor.h"
#include "../src/tokenizer.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using chapter01::DType;
using chapter01::Device;
using chapter01::ModelLoader;
using chapter01::ModelConfig;
using chapter01::ModelMetadata;
using chapter01::MetadataMap;
using chapter01::ModelMetadataAdapter;
using chapter01::GgufReader;
using chapter01::GgufModelIndex;
using chapter01::GgufTensorLoader;
using chapter01::LoadedModel;
using chapter01::ModelBuilder;
using chapter01::TokenizerMetadata;
using chapter01::TokenizerMetadataAdapter;
using chapter01::Tensor;
using chapter01::Tokenizer;

namespace {

constexpr char kMagic[4] = {'C', 'I', 'E', '1'};
constexpr std::uint32_t kVersion = 1;

template <typename T>
void write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_string(std::ostream& out, const std::string& value) {
    const std::uint32_t size = static_cast<std::uint32_t>(value.size());
    write_pod(out, size);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_gguf_string(std::ostream& out, const std::string& value) {
    const auto size = static_cast<std::uint64_t>(value.size());
    write_pod(out, size);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_gguf_kv_string(std::ostream& out, const std::string& key, const std::string& value) {
    write_gguf_string(out, key);
    const std::uint32_t type = 8;
    write_pod(out, type);
    write_gguf_string(out, value);
}

void write_gguf_kv_u32(std::ostream& out, const std::string& key, std::uint32_t value) {
    write_gguf_string(out, key);
    const std::uint32_t type = 4;
    write_pod(out, type);
    write_pod(out, value);
}

template <typename T>
void write_gguf_array(std::ostream& out, const std::string& key, std::uint32_t element_type, const std::vector<T>& values) {
    write_gguf_string(out, key);
    const std::uint32_t array_type = 9;
    write_pod(out, array_type);
    write_pod(out, element_type);
    const auto count = static_cast<std::uint64_t>(values.size());
    write_pod(out, count);
    for (const auto& value : values) {
        if constexpr (std::is_same_v<T, std::string>) {
            write_gguf_string(out, value);
        } else {
            write_pod(out, value);
        }
    }
}

void create_mock_gguf(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    assert(out);
    out.write("GGUF", 4);
    const std::uint32_t version = 3;
    const std::uint64_t tensor_count = 1;
    const std::uint64_t metadata_count = 11;
    write_pod(out, version);
    write_pod(out, tensor_count);
    write_pod(out, metadata_count);
    write_gguf_kv_string(out, "general.architecture", "llama");
    write_gguf_kv_u32(out, "tokenizer.ggml.vocab_size", 64);
    write_gguf_kv_u32(out, "llama.embedding_length", 16);
    write_gguf_kv_u32(out, "llama.context_length", 32);
    write_gguf_kv_u32(out, "llama.block_count", 2);
    write_gguf_kv_u32(out, "tokenizer.ggml.bos_token_id", 1);
    write_gguf_kv_u32(out, "tokenizer.ggml.eos_token_id", 2);
    std::vector<std::string> tokens(64);
    std::vector<float> scores(64, -1.0f);
    std::vector<std::uint32_t> token_types(64, 1);
    tokens[0] = "a";
    tokens[1] = "b";
    tokens[2] = "<eos>";
    token_types[2] = 3;
    write_gguf_array(out, "tokenizer.ggml.tokens", 8, tokens);
    write_gguf_array(out, "tokenizer.ggml.scores", 6, scores);
    write_gguf_array(out, "tokenizer.ggml.token_type", 4, token_types);
    write_gguf_kv_u32(out, "general.alignment", 32);
    write_gguf_string(out, "token_embd.weight");
    const std::uint32_t dimension_count = 2;
    const std::uint64_t dimension0 = 16;
    const std::uint64_t dimension1 = 64;
    const std::uint32_t type = 0;
    const std::uint64_t offset = 0;
    write_pod(out, dimension_count);
    write_pod(out, dimension0);
    write_pod(out, dimension1);
    write_pod(out, type);
    write_pod(out, offset);
    const auto position = static_cast<std::uint64_t>(out.tellp());
    const auto padding = (32 - (position % 32)) % 32;
    for (std::uint64_t i = 0; i < padding; ++i) {
        const char zero = 0;
        out.write(&zero, 1);
    }
    const std::vector<float> payload(64 * 16, 3.5f);
    out.write(reinterpret_cast<const char*>(payload.data()),
              static_cast<std::streamsize>(payload.size() * sizeof(float)));
}

void write_shape(std::ostream& out, const std::vector<std::int64_t>& shape) {
    const std::uint32_t ndim = static_cast<std::uint32_t>(shape.size());
    write_pod(out, ndim);
    for (auto dim : shape) {
        write_pod(out, dim);
    }
}

template <typename T>
std::vector<std::byte> to_bytes(const std::vector<T>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(T));
    if (!bytes.empty()) {
        std::memcpy(bytes.data(), values.data(), bytes.size());
    }
    return bytes;
}

void write_tensor_record(std::ostream& out,
                         const std::string& name,
                         DType dtype,
                         Device device,
                         const std::vector<std::int64_t>& shape,
                         const std::vector<std::byte>& bytes) {
    write_string(out, name);
    const std::uint8_t dtype_u8 = static_cast<std::uint8_t>(dtype);
    const std::uint8_t device_u8 = static_cast<std::uint8_t>(device);
    write_pod(out, dtype_u8);
    write_pod(out, device_u8);
    write_shape(out, shape);
    const std::uint64_t byte_count = static_cast<std::uint64_t>(bytes.size());
    write_pod(out, byte_count);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::filesystem::path make_temp_model_path() {
    auto path = std::filesystem::temp_directory_path() / "chapter01_tensor_loader_mock.bin";
    std::filesystem::remove(path);
    return path;
}

void create_mock_model(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    assert(out);

    out.write(kMagic, sizeof(kMagic));
    write_pod(out, kVersion);

    const std::uint32_t tensor_count = 2;
    write_pod(out, tensor_count);

    {
        const std::vector<float> values = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
        write_tensor_record(out,
                            "linear.weight",
                            DType::Float32,
                            Device::CPU,
                            {2, 3},
                            to_bytes(values));
    }

    {
        const std::vector<std::int8_t> values = {-1, 0, 1, 2};
        write_tensor_record(out,
                            "quant.weight",
                            DType::Int8,
                            Device::CPU,
                            {4},
                            to_bytes(values));
    }
}

void test_tensor_roundtrip() {
    Tensor tensor = Tensor::from_vector<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
    assert(tensor.name() == "x");
    assert(tensor.dtype() == DType::Float32);
    assert(tensor.numel() == 4);
    const auto values = tensor.to_vector<float>();
    assert(values.size() == 4);
    assert(values[0] == 1.f);
    assert(values[3] == 4.f);

    Tensor offset_tensor("offset", {1}, DType::Float32);
    bool rejected_offset = false;
    try {
        offset_tensor.set_storage(std::make_shared<Tensor::Storage>(1), 2);
    } catch (const std::invalid_argument&) {
        rejected_offset = true;
    }
    assert(rejected_offset);

    Tensor oversized("oversized", {std::numeric_limits<std::int64_t>::max(), 3}, DType::Float32);
    bool rejected_shape = false;
    try {
        (void)oversized.numel();
    } catch (const std::overflow_error&) {
        rejected_shape = true;
    }
    assert(rejected_shape);
}

void test_model_loader() {
    const auto path = make_temp_model_path();
    create_mock_model(path);

    ModelLoader loader;
    std::string error;
    const bool ok = loader.load(path, &error);
    assert(ok && error.empty());

    const Tensor* linear = loader.find("linear.weight");
    assert(linear != nullptr);
    assert(linear->dtype() == DType::Float32);
    assert(linear->shape().size() == 2);
    assert(linear->shape()[0] == 2);
    assert(linear->shape()[1] == 3);
    const auto linear_values = linear->to_vector<float>();
    assert(linear_values.size() == 6);
    assert(linear_values[0] == 1.f);
    assert(linear_values[5] == 6.f);

    const Tensor* quant = loader.find("quant.weight");
    assert(quant != nullptr);
    assert(quant->dtype() == DType::Int8);
    assert(quant->shape().size() == 1);
    assert(quant->shape()[0] == 4);
    const auto quant_values = quant->to_vector<std::int8_t>();
    assert((quant_values == std::vector<std::int8_t>{-1, 0, 1, 2}));

    std::filesystem::remove(path);
}

void test_tokenizer_contract() {
    const Tokenizer tokenizer(64);
    const auto tokens = tokenizer.encode("a!");
    assert(tokenizer.vocab_size() == 64);
    assert(tokens.size() == 2);
    assert(tokens[0] == 1);
    assert(tokens[1] == 0);
    assert(tokenizer.decode(tokens) == "aa");
}

void test_model_config_contract() {
    ModelConfig config;
    Tokenizer tokenizer(config.vocab_size);
    std::string error;
    assert(config.valid());
    assert(config.validate_tokenizer(tokenizer, &error));
    assert(error.empty());

    Tensor embedding = Tensor::from_vector<float>(
        "embedding.weight", {64, 16}, std::vector<float>(64 * 16, 1.0f));
    assert(config.validate_tensor(embedding, &error));

    Tensor wrong_embedding = Tensor::from_vector<float>(
        "embedding.weight", {32, 16}, std::vector<float>(32 * 16, 1.0f));
    assert(!config.validate_tensor(wrong_embedding, &error));
    assert(!error.empty());
}

void test_model_metadata_boundary() {
    ModelMetadata metadata;
    metadata.architecture = "llama";
    metadata.vocab_size = 64;
    metadata.hidden_size = 16;
    metadata.max_position = 32;
    metadata.layer_count = 2;
    metadata.bos_token_id = 1;
    metadata.eos_token_id = 2;

    assert(metadata.valid());
    const ModelConfig config = ModelConfig::from_metadata(metadata);
    assert(config.architecture == "llama");
    assert(config.vocab_size == 64);
    assert(config.hidden_size == 16);
    assert(config.bos_token_id == 1);
    assert(config.eos_token_id == 2);
}

void test_gguf_metadata_adapter() {
    const MetadataMap values = {
        {"general.architecture", std::string("llama")},
        {"tokenizer.ggml.vocab_size", std::uint64_t{32000}},
        {"llama.embedding_length", std::uint64_t{2048}},
        {"llama.context_length", std::uint64_t{2048}},
        {"llama.block_count", std::uint64_t{22}},
        {"tokenizer.ggml.bos_token_id", std::int64_t{1}},
        {"tokenizer.ggml.eos_token_id", std::int64_t{2}},
    };

    ModelMetadata metadata;
    std::string error;
    assert(ModelMetadataAdapter::from_gguf(values, &metadata, &error));
    assert(error.empty());
    assert(metadata.architecture == "llama");
    assert(metadata.vocab_size == 32000);
    assert(metadata.hidden_size == 2048);
    assert(metadata.max_position == 2048);
    assert(metadata.layer_count == 22);
    assert(metadata.bos_token_id == 1);
    assert(metadata.eos_token_id == 2);

    MetadataMap incomplete = values;
    incomplete.erase("llama.block_count");
    assert(!ModelMetadataAdapter::from_gguf(incomplete, &metadata, &error));
    assert(error.find("block_count") != std::string::npos);

    MetadataMap inferred_vocab = values;
    inferred_vocab.erase("tokenizer.ggml.vocab_size");
    inferred_vocab["tokenizer.ggml.tokens"] = std::vector<std::string>(64, "token");
    assert(ModelMetadataAdapter::from_gguf(inferred_vocab, &metadata, &error));
    assert(metadata.vocab_size == 64);

    MetadataMap signed_token_types = values;
    signed_token_types["tokenizer.ggml.tokens"] = std::vector<std::string>(64, "token");
    signed_token_types["tokenizer.ggml.scores"] = std::vector<float>(64, -1.0f);
    signed_token_types["tokenizer.ggml.token_type"] = std::vector<std::int32_t>(64, 1);
    TokenizerMetadata tokenizer_metadata;
    assert(TokenizerMetadataAdapter::from_gguf(signed_token_types, &tokenizer_metadata, &error));
}

void test_gguf_reader() {
    const auto path = std::filesystem::temp_directory_path() / "chapter01_mock.gguf";
    std::filesystem::remove(path);
    create_mock_gguf(path);

    GgufModelIndex index;
    std::string error;
    const bool index_loaded = GgufReader::read_index(path.string(), &index, &error);
    if (!index_loaded) throw std::runtime_error(error);
    assert(index_loaded);
    assert(error.empty());
    assert(index.version == 3);
    assert(std::get<std::string>(index.metadata.at("general.architecture")) == "llama");
    assert(index.tensors.size() == 1);
    assert(index.tensors[0].name == "token_embd.weight");
    assert(index.tensors[0].dimensions == std::vector<std::uint64_t>({16, 64}));
    assert(index.tensors[0].type == 0);
    assert(index.tensors[0].offset == 0);
    assert(index.data_offset % 32 == 0);

    ModelMetadata metadata;
    assert(ModelMetadataAdapter::from_gguf(index.metadata, &metadata, &error));
    assert(metadata.valid());
    assert(metadata.architecture == "llama");
    assert(metadata.vocab_size == 64);
    assert(metadata.hidden_size == 16);
    assert(metadata.max_position == 32);
    assert(metadata.layer_count == 2);
    assert(metadata.bos_token_id == 1);
    assert(metadata.eos_token_id == 2);

    TokenizerMetadata tokenizer_metadata;
    assert(TokenizerMetadataAdapter::from_gguf(index.metadata, &tokenizer_metadata, &error));
    assert(tokenizer_metadata.valid());
    assert(tokenizer_metadata.tokens.size() == 64);
    assert(tokenizer_metadata.tokens[2] == "<eos>");
    assert(tokenizer_metadata.scores.size() == 64);
    assert(tokenizer_metadata.token_types.size() == 64);

    std::unordered_map<std::string, Tensor> tensors;
    const bool tensors_loaded = GgufTensorLoader::load_all(path, index, &tensors, &error);
    if (!tensors_loaded) throw std::runtime_error(error);
    assert(tensors_loaded);
    const Tensor* embedding = tensors.at("token_embd.weight").name().empty()
                                  ? nullptr
                                  : &tensors.at("token_embd.weight");
    assert(embedding != nullptr);
    assert(embedding->dtype() == DType::Float32);
    assert(embedding->bytes() == 64 * 16 * sizeof(float));
    assert(embedding->to_vector<float>()[0] == 3.5f);

    auto add_tensor = [&tensors](const std::string& name, std::vector<std::int64_t> shape) {
        std::size_t elements = 1;
        for (const auto dimension : shape) elements *= static_cast<std::size_t>(dimension);
        tensors.emplace(name, Tensor::from_vector<float>(name, std::move(shape), std::vector<float>(elements, 1.0f)));
    };
    add_tensor("output.weight", {16, 64});
    add_tensor("output_norm.weight", {16});
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        add_tensor(prefix + "attn_norm.weight", {16});
        add_tensor(prefix + "attn_q.weight", {16, 16});
        add_tensor(prefix + "attn_k.weight", {16, 8});
        add_tensor(prefix + "attn_v.weight", {16, 8});
        add_tensor(prefix + "attn_output.weight", {16, 16});
        add_tensor(prefix + "ffn_norm.weight", {16});
        add_tensor(prefix + "ffn_gate.weight", {16, 32});
        add_tensor(prefix + "ffn_up.weight", {16, 32});
        add_tensor(prefix + "ffn_down.weight", {32, 16});
    }

    LoadedModel model;
    std::unordered_map<std::string, Tensor> incomplete_tensors = tensors;
    incomplete_tensors.erase("output.weight");
    assert(!ModelBuilder::build(metadata, tokenizer_metadata, std::move(incomplete_tensors), &model, &error));
    assert(error.find("missing Llama tensor") != std::string::npos);
    assert(ModelBuilder::build(metadata, tokenizer_metadata, std::move(tensors), &model, &error));
    assert(error.empty());
    assert(model.config.architecture == "llama");
    assert(model.tokenizer.tokens.size() == 64);
    assert(model.layout.layers.size() == 2);
    assert(model.layout.intermediate_size == 32);
    assert(model.layout.key_value_size == 8);
    assert(model.decoder.valid());
    assert(model.decoder.embedding == "token_embd.weight");
    assert(model.decoder.layers.size() == 2);
    assert(model.decoder.layers[0].attn_q == "blk.0.attn_q.weight");
    assert(model.decoder.layers[1].ffn_down == "blk.1.ffn_down.weight");
    assert(model.tensors.size() == 21);
    std::filesystem::remove(path);
}

void test_q4_k_payload_loader() {
    const auto path = std::filesystem::temp_directory_path() / "chapter01_q4_k.gguf";
    std::filesystem::remove(path);
    {
        std::ofstream out(path, std::ios::binary);
        assert(out);
        std::vector<std::byte> payload(144, std::byte{0x2a});
        out.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
    }

    GgufModelIndex index;
    index.data_offset = 0;
    index.tensors.push_back({"q4.weight", {256}, 12, 0});
    std::unordered_map<std::string, Tensor> tensors;
    std::string error;
    const bool loaded = GgufTensorLoader::load_all(path, index, &tensors, &error);
    if (!loaded) throw std::runtime_error(error);
    assert(loaded);
    const Tensor& tensor = tensors.at("q4.weight");
    assert(tensor.dtype() == DType::Q4_K);
    assert(tensor.numel() == 256);
    assert(tensor.bytes() == 144);
    assert(tensor.raw_data()[0] == std::byte{0x2a});
    std::filesystem::remove(path);
}

void test_q8_0_payload_loader() {
    const auto path = std::filesystem::temp_directory_path() / "chapter01_q8_0.gguf";
    std::filesystem::remove(path);
    {
        std::ofstream out(path, std::ios::binary);
        assert(out);
        std::vector<std::byte> payload(34, std::byte{0x17});
        out.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
    }

    GgufModelIndex index;
    index.data_offset = 0;
    index.tensors.push_back({"q8.weight", {32}, 8, 0});
    Tensor tensor;
    std::string error;
    const bool loaded = GgufTensorLoader::load_tensor(
        path, index, index.tensors.front(), &tensor, &error);
    if (!loaded) throw std::runtime_error(error);
    assert(loaded);
    assert(error.empty());
    assert(tensor.dtype() == DType::Q8_0);
    assert(tensor.numel() == 32);
    assert(tensor.bytes() == 34);
    assert(tensor.raw_data()[33] == std::byte{0x17});

    index.tensors.front().dimensions = {31};
    const bool rejected = !GgufTensorLoader::load_tensor(
        path, index, index.tensors.front(), &tensor, &error);
    if (!rejected) throw std::runtime_error("misaligned Q8_0 tensor was accepted");
    assert(rejected);
    assert(error == "GGUF quantized tensor shape is not block aligned");
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    test_tensor_roundtrip();
    test_model_loader();
    test_tokenizer_contract();
    test_model_config_contract();
    test_model_metadata_boundary();
    test_gguf_metadata_adapter();
    test_gguf_reader();
    test_q8_0_payload_loader();
    test_q4_k_payload_loader();
    std::cout << "chapter01_tensor_loader tests passed\n";
    return 0;
}
