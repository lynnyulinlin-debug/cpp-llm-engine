#include <cassert>
#include <cstddef>
#include <memory>

class Buffer {
 public:
  explicit Buffer(std::size_t size) : size_(size), data_(size ? std::make_unique<float[]>(size) : nullptr) {}
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept : size_(other.size_), data_(std::move(other.data_)) { other.size_ = 0; }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      size_ = other.size_;
      other.size_ = 0;
    }
    return *this;
  }
  std::size_t size() const { return size_; }
  float* data() { return data_.get(); }

 private:
  std::size_t size_ = 0;
  std::unique_ptr<float[]> data_;
};

int main() {
  Buffer source(4);
  source.data()[0] = 7.0F;
  Buffer target(std::move(source));
  assert(source.size() == 0);
  assert(target.size() == 4);
  assert(target.data()[0] == 7.0F);

  Buffer replacement(1);
  replacement = std::move(target);
  assert(target.size() == 0);
  assert(replacement.size() == 4);
  assert(replacement.data()[0] == 7.0F);
}
