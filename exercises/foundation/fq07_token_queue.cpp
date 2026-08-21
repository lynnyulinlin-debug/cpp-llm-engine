#include <cassert>
#include <cstddef>
#include <deque>
#include <stdexcept>

class TokenQueue {
 public:
  void push(int token) { tokens_.push_back(token); }
  int pop() {
    if (tokens_.empty()) throw std::out_of_range("token queue is empty");
    const int token = tokens_.front();
    tokens_.pop_front();
    return token;
  }
  std::size_t size() const { return tokens_.size(); }

 private:
  std::deque<int> tokens_;
};

int main() {
  TokenQueue queue;
  queue.push(11);
  queue.push(12);
  assert(queue.size() == 2);
  assert(queue.pop() == 11);
  assert(queue.pop() == 12);
  bool rejected = false;
  try { (void)queue.pop(); } catch (const std::out_of_range&) { rejected = true; }
  assert(rejected);
}
