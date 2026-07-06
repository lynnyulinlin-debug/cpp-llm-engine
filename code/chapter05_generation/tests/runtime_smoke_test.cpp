#include "chapter05_generation/request.h"

#include <cassert>

int main() {
    chapter05_generation::Request request({1, 2, 3});
    assert(request.prompt().size() == 3);
    assert(request.prompt_size() == 3);
    assert(request.generated_size() == 0);
    assert(!request.finished());
    assert(request.phase() == chapter05_generation::RequestPhase::Initial);
    return 0;
}
