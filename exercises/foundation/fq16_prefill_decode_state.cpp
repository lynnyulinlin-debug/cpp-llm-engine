#include <cassert>
#include <cstddef>
#include <stdexcept>

enum class Phase { Waiting, Prefill, Decode, Finished, Aborted };

class GenerationState {
 public:
  explicit GenerationState(std::size_t max_new_tokens)
      : max_new_tokens_(max_new_tokens) {
    if (max_new_tokens == 0) throw std::invalid_argument("generation limit must not be zero");
  }

  Phase phase() const { return phase_; }
  std::size_t generated_tokens() const { return generated_tokens_; }

  void admit() {
    require(Phase::Waiting);
    phase_ = Phase::Prefill;
  }

  void finish_prefill() {
    require(Phase::Prefill);
    phase_ = Phase::Decode;
  }

  bool append_decode_token() {
    require(Phase::Decode);
    ++generated_tokens_;
    if (generated_tokens_ == max_new_tokens_) phase_ = Phase::Finished;
    return phase_ == Phase::Finished;
  }

  void abort() {
    if (phase_ == Phase::Finished || phase_ == Phase::Aborted) {
      throw std::logic_error("request is already terminal");
    }
    phase_ = Phase::Aborted;
  }

 private:
  void require(Phase expected) const {
    if (phase_ != expected) throw std::logic_error("invalid generation transition");
  }

  std::size_t max_new_tokens_;
  std::size_t generated_tokens_ = 0;
  Phase phase_ = Phase::Waiting;
};

int main() {
  GenerationState request(2);
  assert(request.phase() == Phase::Waiting);
  request.admit();
  assert(request.phase() == Phase::Prefill);
  request.finish_prefill();
  assert(request.phase() == Phase::Decode);
  assert(!request.append_decode_token());
  assert(request.append_decode_token());
  assert(request.phase() == Phase::Finished);

  GenerationState cancelled(3);
  cancelled.admit();
  cancelled.abort();
  assert(cancelled.phase() == Phase::Aborted);

  bool rejected = false;
  try {
    cancelled.finish_prefill();
  } catch (const std::logic_error&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
