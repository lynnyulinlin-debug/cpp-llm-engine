#include "chapter05_generation/runtime.h"
#include "chapter05_generation/transformer.h"
#include "chapter05_generation/sampler.h"

#include <cassert>

int main() {
    chapter05_generation::GenerationConfig config;
    config.max_new_tokens = 3;
    config.use_kv_cache = true;

    chapter05_generation::Request request({1, 2, 3});
    chapter05_generation::GreedySampler sampler;
    chapter05_generation::Embedding embedding(32);
    chapter05_generation::PositionEncoding position_encoding(16);
    chapter05_generation::Transformer transformer(2, embedding, position_encoding);
    chapter05_generation::Runtime runtime(config, transformer);

    assert(embedding.ready());
    assert(position_encoding.ready());
    assert(transformer.ready());
    assert(transformer.layer_count() == 2);
    assert(transformer.embedding().vocab_size() == 32);
    assert(transformer.position_encoding().max_length() == 16);

    const std::size_t prefills = runtime.prefill(request);
    assert(prefills == 3);
    assert(request.prefilled());
    assert(request.phase() == chapter05_generation::RequestPhase::Prefilled);
    assert(runtime.cache().size() == 3);

    const std::size_t generated_steps = runtime.decode(request, sampler);
    assert(generated_steps == 3);
    assert(request.generated_size() == 3);
    assert(request.phase() == chapter05_generation::RequestPhase::Finished);
    assert(runtime.cache().size() == 6);
    assert(request.finished());
    return 0;
}
