#include "chapter05_generation/kv_cache.h"

#include <cassert>

int main() {
    chapter05_generation::KVCache cache;
    cache.append(1);
    cache.append(2);
    assert(cache.size() == 2);
    cache.clear();
    assert(cache.size() == 0);
    return 0;
}
