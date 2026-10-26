#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

class Workspace {
 public:
  explicit Workspace(std::size_t size) : data_(size ? std::make_unique<int[]>(size) : nullptr), size_(size) {}

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  Workspace(Workspace&& other) noexcept
      : data_(std::move(other.data_)), size_(other.size_) {
    other.size_ = 0;
  }

  Workspace& operator=(Workspace&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      size_ = other.size_;
      other.size_ = 0;
    }
    return *this;
  }

  int& at(std::size_t index) {
    assert(index < size_);
    return data_[index];
  }

  std::size_t size() const { return size_; }

 private:
  std::unique_ptr<int[]> data_;
  std::size_t size_ = 0;
};

class EngineStep {
 public:
  explicit EngineStep(std::size_t workspace_size) : workspace_(workspace_size) {}

  int run(int input) {
    workspace_.at(0) = input;
    return workspace_.at(0) + 1;
  }

 private:
  Workspace workspace_;
};

int main() {
  EngineStep step(1);
  assert(step.run(41) == 42);

  Workspace first(2);
  Workspace second = std::move(first);
  assert(first.size() == 0);
  assert(second.size() == 2);
  second.at(1) = 7;
  assert(second.at(1) == 7);
  return 0;
}
