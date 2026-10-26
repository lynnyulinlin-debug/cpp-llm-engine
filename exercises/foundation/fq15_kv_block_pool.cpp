#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

class KvBlockPool {
 public:
  explicit KvBlockPool(std::size_t block_count) : free_(block_count, true) {
    if (block_count == 0) throw std::invalid_argument("block pool must not be empty");
  }

  bool allocate(int request_id, std::size_t block_count) {
    if (block_count == 0 || owners_.count(request_id) != 0 || free_count() < block_count) {
      return false;
    }

    std::vector<std::size_t> blocks;
    for (std::size_t index = 0; index < free_.size() && blocks.size() < block_count; ++index) {
      if (free_[index]) {
        free_[index] = false;
        blocks.push_back(index);
      }
    }
    owners_.emplace(request_id, std::move(blocks));
    return true;
  }

  bool release(int request_id) {
    const auto it = owners_.find(request_id);
    if (it == owners_.end()) return false;
    for (std::size_t block : it->second) free_[block] = true;
    owners_.erase(it);
    return true;
  }

  std::size_t free_count() const {
    std::size_t result = 0;
    for (bool free : free_) {
      if (free) ++result;
    }
    return result;
  }

  std::size_t blocks_for(int request_id) const {
    const auto it = owners_.find(request_id);
    return it == owners_.end() ? 0 : it->second.size();
  }

 private:
  std::vector<bool> free_;
  std::unordered_map<int, std::vector<std::size_t>> owners_;
};

int main() {
  KvBlockPool pool(4);
  assert(pool.free_count() == 4);
  assert(pool.allocate(10, 2));
  assert(pool.blocks_for(10) == 2);
  assert(!pool.allocate(10, 1));
  assert(!pool.allocate(11, 3));
  assert(pool.allocate(11, 2));
  assert(pool.free_count() == 0);
  assert(!pool.release(99));
  assert(pool.release(10));
  assert(pool.free_count() == 2);
  assert(pool.allocate(12, 1));
  assert(pool.blocks_for(12) == 1);
  assert(pool.release(11));
  assert(pool.release(12));
  assert(pool.free_count() == 4);
  return 0;
}
