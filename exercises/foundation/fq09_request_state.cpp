#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

enum class RequestPhase { Waiting, Prefill, Running, Finished, Aborted };

class RequestState {
 public:
  RequestState(std::size_t prompt_size, std::size_t max_new_tokens)
      : prompt_size_(prompt_size), max_new_tokens_(max_new_tokens) {}

  RequestPhase phase() const { return phase_; }
  std::size_t generated_size() const { return generated_.size(); }

  void admit() {
    require(RequestPhase::Waiting);
    phase_ = RequestPhase::Prefill;
  }

  void finish_prefill() {
    require(RequestPhase::Prefill);
    phase_ = RequestPhase::Running;
  }

  bool append_generated(int token) {
    require(RequestPhase::Running);
    generated_.push_back(token);
    if (generated_.size() >= max_new_tokens_) phase_ = RequestPhase::Finished;
    return phase_ == RequestPhase::Finished;
  }

  void abort() {
    if (phase_ == RequestPhase::Finished || phase_ == RequestPhase::Aborted) {
      throw std::logic_error("request is already terminal");
    }
    phase_ = RequestPhase::Aborted;
  }

 private:
  void require(RequestPhase expected) const {
    if (phase_ != expected) throw std::logic_error("invalid request transition");
  }

  std::size_t prompt_size_ = 0;
  std::size_t max_new_tokens_ = 0;
  std::vector<int> generated_;
  RequestPhase phase_ = RequestPhase::Waiting;
};

int main() {
  RequestState request(3, 2);
  assert(request.phase() == RequestPhase::Waiting);
  request.admit();
  request.finish_prefill();
  assert(!request.append_generated(7));
  assert(request.append_generated(8));
  assert(request.phase() == RequestPhase::Finished);
  assert(request.generated_size() == 2);

  RequestState cancelled(1, 4);
  cancelled.admit();
  cancelled.abort();
  assert(cancelled.phase() == RequestPhase::Aborted);

  bool rejected = false;
  try {
    cancelled.finish_prefill();
  } catch (const std::logic_error&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
