#include "tensor.h"

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace chapter01 {

namespace {

std::size_t product(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    std::size_t n = 1;
    for (auto dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("tensor shape cannot contain negative dimensions");
        }
        n *= static_cast<std::size_t>(dim);
    }
    return n;
}

}  // namespace

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return sizeof(float);
        case DType::Int8:
            return sizeof(std::int8_t);
        case DType::Unknown:
        default:
            return 0;
    }
}

Tensor::Tensor() : storage_(std::make_shared<Storage>()) {}

Tensor::Tensor(std::string name, std::vector<std::int64_t> shape, DType dtype, Device device)
    : name_(std::move(name)),
      shape_(std::move(shape)),
      dtype_(dtype),
      device_(device),
      storage_(std::make_shared<Storage>()) {}

Tensor Tensor::from_raw_bytes(std::string name,
                              std::vector<std::int64_t> shape,
                              DType dtype,
                              std::vector<std::byte> bytes,
                              Device device) {
    Tensor tensor(std::move(name), std::move(shape), dtype, device);
    tensor.set_storage(std::make_shared<Storage>(std::move(bytes)), 0);
    return tensor;
}

const std::string& Tensor::name() const { return name_; }

void Tensor::set_name(std::string name) { name_ = std::move(name); }

const std::vector<std::int64_t>& Tensor::shape() const { return shape_; }

void Tensor::set_shape(std::vector<std::int64_t> shape) { shape_ = std::move(shape); }

DType Tensor::dtype() const { return dtype_; }

void Tensor::set_dtype(DType dtype) { dtype_ = dtype; }

Device Tensor::device() const { return device_; }

void Tensor::set_device(Device device) { device_ = device; }

std::size_t Tensor::numel() const { return product(shape_); }

std::size_t Tensor::bytes() const { return numel() * dtype_size(dtype_); }

const std::byte* Tensor::raw_data() const {
    if (!storage_) {
        return nullptr;
    }
    return storage_->data() + offset_bytes_;
}

std::byte* Tensor::raw_data() {
    if (!storage_) {
        return nullptr;
    }
    return storage_->data() + offset_bytes_;
}

const Tensor::StoragePtr& Tensor::storage() const { return storage_; }

void Tensor::set_storage(StoragePtr storage, std::size_t offset_bytes) {
    storage_ = std::move(storage);
    offset_bytes_ = offset_bytes;
}

std::size_t Tensor::offset_bytes() const { return offset_bytes_; }

void Tensor::set_offset_bytes(std::size_t offset_bytes) { offset_bytes_ = offset_bytes; }

}  // namespace chapter01
