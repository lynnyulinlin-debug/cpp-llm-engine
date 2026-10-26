// Chapter 05：测试 request-owned KV Cache 的 append、clear 和 token-history semantics。
// 契约：这些用例定义 Generation Core 在 block cache 之前消费的 cache contract。
// 验证入口：Chapter 05 CTest target 或 scripts/run.sh。
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
