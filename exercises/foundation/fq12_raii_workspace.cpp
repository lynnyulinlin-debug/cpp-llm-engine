#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

class Workspace {
 public:
  explicit Workspace(std::size_t size)
      : storage_(size == 0 ? nullptr : std::make_unique<float[]>(size)),
        size_(size) {
    if (size == 0) throw std::invalid_argument("workspace must not be empty");
  }

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  Workspace(Workspace&&) noexcept = default;
  Workspace& operator=(Workspace&&) noexcept = default;

  std::size_t size() const { return size_; }
  bool valid() const { return storage_ != nullptr; }

  float& at(std::size_t index) {
    if (index >= size_) throw std::out_of_range("workspace index");
    return storage_[index];
  }

 private:
  std::unique_ptr<float[]> storage_;
  std::size_t size_ = 0;
};

int main() {
  Workspace workspace(2);
  workspace.at(0) = 1.0f;
  workspace.at(1) = 2.0f;

  Workspace moved(std::move(workspace));
  assert(!workspace.valid());
  assert(moved.size() == 2);
  assert(moved.at(0) == 1.0f);
  assert(moved.at(1) == 2.0f);

  bool rejected = false;
  try {
    Workspace empty(0);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
