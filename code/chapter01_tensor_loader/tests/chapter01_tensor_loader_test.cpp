#include "../src/model_loader.h"
#include "../src/tensor.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using chapter01::DType;
using chapter01::Device;
using chapter01::ModelLoader;
using chapter01::Tensor;

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

}  // namespace

int main() {
    test_tensor_roundtrip();
    test_model_loader();
    std::cout << "chapter01_tensor_loader tests passed\n";
    return 0;
}
