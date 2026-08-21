#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>

class MiniTensor {
 public:
  explicit MiniTensor(std::size_t size)
      : storage_(std::shared_ptr<float[]>(new float[size], std::default_delete<float[]>())), size_(size) {}
  MiniTensor(std::shared_ptr<float[]> storage, std::size_t size, std::size_t offset)
      : storage_(std::move(storage)), size_(size), offset_(offset) {}
  float& at(std::size_t index) {
    if (index >= size_) throw std::out_of_range("tensor index out of range");
    return storage_[offset_ + index];
  }
  MiniTensor slice(std::size_t offset, std::size_t size) const {
    if (offset + size > size_) throw std::out_of_range("slice out of range");
    return MiniTensor(storage_, size, offset_ + offset);
  }

 private:
  std::shared_ptr<float[]> storage_;
  std::size_t size_ = 0;
  std::size_t offset_ = 0;
};

int main() {
  MiniTensor tensor(4);
  tensor.at(1) = 9.0F;
  auto view = tensor.slice(1, 2);
  assert(view.at(0) == 9.0F);
  view.at(1) = 10.0F;
  assert(tensor.at(2) == 10.0F);
}
