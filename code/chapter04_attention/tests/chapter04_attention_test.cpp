#include "chapter04_attention/decoder_layer.h"

#include <cassert>
#include <cmath>

namespace {

chapter04_attention::Tensor make_input() {
    chapter04_attention::Tensor input(2, 4);
    input.at(0, 0) = 1.0f;
    input.at(0, 1) = 2.0f;
    input.at(0, 2) = 3.0f;
    input.at(0, 3) = 4.0f;
    input.at(1, 0) = 5.0f;
    input.at(1, 1) = 6.0f;
    input.at(1, 2) = 7.0f;
    input.at(1, 3) = 8.0f;
    return input;
}

}  // namespace

int main() {
    const chapter04_attention::Tensor input = make_input();

    chapter04_attention::LayerNorm norm(4);
    const chapter04_attention::Tensor normed = norm.forward(input);
    assert(norm.ready());
    assert(normed.ready());
    assert(normed.rows == 2);
    assert(normed.cols == 4);

    chapter04_attention::MultiHeadAttention attention(4, 2);
    const chapter04_attention::Tensor attended = attention.forward(input);
    assert(attention.ready());
    assert(attended.ready());
    assert(attended.rows == 2);
    assert(attended.cols == 4);

    chapter04_attention::FeedForward feed_forward(4, 8);
    const chapter04_attention::Tensor ffn_output = feed_forward.forward(input);
    assert(feed_forward.ready());
    assert(ffn_output.ready());
    assert(ffn_output.rows == 2);
    assert(ffn_output.cols == 4);

    chapter04_attention::DecoderLayer layer(4, 2, 8, 0);
    const chapter04_attention::Tensor output = layer.forward(input);
    assert(layer.ready());
    assert(output.ready());
    assert(output.rows == 2);
    assert(output.cols == 4);
    assert(std::abs(output.at(0, 0) - input.at(0, 0)) > 1e-4f);
    assert(std::abs(output.at(1, 3) - input.at(1, 3)) > 1e-4f);

    return 0;
}
