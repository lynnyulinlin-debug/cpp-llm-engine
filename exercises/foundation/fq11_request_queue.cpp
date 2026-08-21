#include <cassert>
#include <cstddef>
#include <deque>
#include <optional>

class RequestQueue {
 public:
  explicit RequestQueue(std::size_t capacity) : capacity_(capacity) {}

  bool admit(int request_id) {
    if (queue_.size() >= capacity_) return false;
    queue_.push_back(request_id);
    return true;
  }

  bool cancel(int request_id) {
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
      if (*it == request_id) {
        queue_.erase(it);
        return true;
      }
    }
    return false;
  }

  std::optional<int> pop() {
    if (queue_.empty()) return std::nullopt;
    const int request_id = queue_.front();
    queue_.pop_front();
    return request_id;
  }

  std::size_t size() const { return queue_.size(); }

 private:
  std::size_t capacity_ = 0;
  std::deque<int> queue_;
};

int main() {
  RequestQueue queue(2);
  assert(queue.admit(10));
  assert(queue.admit(11));
  assert(!queue.admit(12));
  assert(queue.cancel(10));
  assert(queue.admit(12));
  assert(queue.pop() == 11);
  assert(queue.pop() == 12);
  assert(!queue.pop().has_value());
  assert(!queue.cancel(99));
  return 0;
}
