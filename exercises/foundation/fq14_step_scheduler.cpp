#include <cassert>
#include <cstddef>
#include <deque>
#include <vector>

struct StepResult {
  std::vector<int> request_ids;
  std::size_t tokens = 0;
};

class StepScheduler {
 public:
  void admit(int request_id, std::size_t remaining_tokens) {
    if (remaining_tokens == 0) return;
    queue_.push_back({request_id, remaining_tokens});
  }

  StepResult run_step(std::size_t max_batch, std::size_t token_budget) {
    StepResult result;
    if (max_batch == 0 || token_budget == 0) return result;

    const std::size_t selected = queue_.size();
    for (std::size_t i = 0; i < selected && result.request_ids.size() < max_batch &&
                              result.tokens < token_budget;
         ++i) {
      Entry entry = queue_.front();
      queue_.pop_front();
      result.request_ids.push_back(entry.id);
      ++result.tokens;
      --entry.remaining_tokens;
      if (entry.remaining_tokens > 0) queue_.push_back(entry);
    }
    return result;
  }

  bool empty() const { return queue_.empty(); }

 private:
  struct Entry {
    int id;
    std::size_t remaining_tokens;
  };
  std::deque<Entry> queue_;
};

int main() {
  StepScheduler scheduler;
  scheduler.admit(10, 2);
  scheduler.admit(11, 1);
  scheduler.admit(12, 2);

  StepResult first = scheduler.run_step(2, 2);
  assert((first.request_ids == std::vector<int>{10, 11}));
  assert(first.tokens == 2);

  StepResult second = scheduler.run_step(2, 2);
  assert((second.request_ids == std::vector<int>{12, 10}));
  assert(second.tokens == 2);

  StepResult third = scheduler.run_step(2, 2);
  assert((third.request_ids == std::vector<int>{12}));
  assert(third.tokens == 1);
  assert(scheduler.empty());
  return 0;
}
