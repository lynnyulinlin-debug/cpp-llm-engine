// Chapter 05：smoke-test Runtime readiness、prefill/decode progression 和 terminal completion。
// 职责：作为 Generation Core contract 的最小可执行证明。
// 验证入口：Chapter 05 CTest target 或 scripts/run.sh。
#include "chapter05_generation/request.h"

#include <cassert>

int main() {
    chapter05_generation::Request request({1, 2, 3});
    assert(request.prompt().size() == 3);
    assert(request.prompt_size() == 3);
    assert(request.generated_size() == 0);
    assert(!request.finished());
    assert(request.phase() == chapter05_generation::RequestPhase::Initial);

    request.cancel();
    assert(request.cancelled());
    assert(request.terminal());
    assert(chapter05_generation::request_phase_name(request.phase()) == "cancelled");

    chapter05_generation::Request timed_out({4, 5});
    timed_out.mark_timed_out();
    assert(timed_out.timed_out());
    assert(timed_out.terminal());

    request.cache().append(9);
    assert(request.cache().size() == 1);
    return 0;
}
