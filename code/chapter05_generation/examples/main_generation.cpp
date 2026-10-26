// Chapter 05：运行非 streaming 的 Generation Core 演示。
// 输出：Request phase、generated step 和 request-owned cache state。
// 验证：使用 scripts/run.sh 执行可重复构建、测试和输出检查。
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/transformer.h"
#include "chapter05_generation/sampler.h"

#include <iostream>

int main() {
    chapter05_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.use_kv_cache = true;

    chapter05_generation::Embedding embedding(16);
    chapter05_generation::PositionEncoding position_encoding(8);
    chapter05_generation::Transformer transformer(2, embedding, position_encoding);
    chapter05_generation::Runtime runtime(config, transformer);
    chapter05_generation::Request request({1, 2, 3});
    chapter05_generation::GreedySampler sampler;

    const std::size_t prefills = runtime.prefill(request);
    const std::size_t generated = runtime.decode(request, sampler);
    std::cout << "[chapter05] generation runtime skeleton\n";
    std::cout << "[chapter05] ready: " << std::boolalpha << runtime.ready() << "\n";
    std::cout << "[chapter05] prefills: " << prefills << "\n";
    std::cout << "[chapter05] generated steps: " << generated << "\n";
    std::cout << "[chapter05] request cache size: " << request.cache().size() << "\n";
    std::cout << "[chapter05] request prefilled: " << std::boolalpha << request.prefilled() << "\n";
    std::cout << "[chapter05] request phase: " << chapter05_generation::request_phase_name(request.phase()) << "\n";
    std::cout << "[chapter05] request finished: " << std::boolalpha << request.finished() << "\n";
    return 0;
}
