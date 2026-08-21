#include <cassert>
#include <cstddef>
#include <vector>

struct TensorView {
  float* data = nullptr;
  std::size_t length = 0;
  std::size_t offset = 0;

  float& at(std::size_t index) {
    assert(index < length);
    return data[offset + index];
  }
};

int main() {
  std::vector<float> storage{0, 1, 2, 3, 4, 5};
  TensorView view{storage.data(), 3, 2};
  view.at(0) = 20;
  view.at(2) = 40;
  assert(storage[2] == 20);
  assert(storage[4] == 40);
}
