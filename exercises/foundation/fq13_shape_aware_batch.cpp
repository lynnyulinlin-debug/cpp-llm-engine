#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

class BatchLayout {
 public:
  explicit BatchLayout(std::size_t capacity) : capacity_(capacity) {
    if (capacity == 0) throw std::invalid_argument("batch capacity must not be zero");
  }

  bool add(std::size_t sequence_length) {
    if (sequence_length == 0 || lengths_.size() == capacity_) return false;
    lengths_.push_back(sequence_length);
    return true;
  }

  std::size_t batch_size() const { return lengths_.size(); }

  std::size_t max_sequence_length() const {
    std::size_t result = 0;
    for (std::size_t length : lengths_) {
      if (length > result) result = length;
    }
    return result;
  }

  std::size_t padded_tokens() const {
    return batch_size() * max_sequence_length();
  }

  std::size_t active_tokens() const {
    std::size_t result = 0;
    for (std::size_t length : lengths_) result += length;
    return result;
  }

  std::size_t row_offset(std::size_t row) const {
    if (row >= batch_size()) throw std::out_of_range("batch row");
    return row * max_sequence_length();
  }

  std::size_t valid_tokens_in_row(std::size_t row) const {
    if (row >= batch_size()) throw std::out_of_range("batch row");
    return lengths_[row];
  }

 private:
  std::size_t capacity_;
  std::vector<std::size_t> lengths_;
};

int main() {
  BatchLayout batch(3);
  assert(batch.add(5));
  assert(batch.add(2));
  assert(batch.add(4));
  assert(!batch.add(3));
  assert(!batch.add(0));
  assert(batch.batch_size() == 3);
  assert(batch.max_sequence_length() == 5);
  assert(batch.active_tokens() == 11);
  assert(batch.padded_tokens() == 15);
  assert(batch.row_offset(0) == 0);
  assert(batch.row_offset(1) == 5);
  assert(batch.row_offset(2) == 10);
  assert(batch.valid_tokens_in_row(1) == 2);

  bool rejected = false;
  try {
    (void)batch.row_offset(3);
  } catch (const std::out_of_range&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
