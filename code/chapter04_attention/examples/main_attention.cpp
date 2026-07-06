#include "chapter04_attention/decoder_layer.h"

#include <iostream>

int main() {
    chapter04_attention::Tensor input(2, 4);
    input.at(0, 0) = 1.0f;
    input.at(0, 1) = 2.0f;
    input.at(0, 2) = 3.0f;
    input.at(0, 3) = 4.0f;
    input.at(1, 0) = 5.0f;
    input.at(1, 1) = 6.0f;
    input.at(1, 2) = 7.0f;
    input.at(1, 3) = 8.0f;

    chapter04_attention::DecoderLayer layer(4, 2, 8, 0);
    const chapter04_attention::Tensor output = layer.forward(input);

    std::cout << "[chapter04] attention demo\n";
    std::cout << "ready: " << std::boolalpha << layer.ready() << '\n';
    std::cout << "output shape: " << output.rows << "x" << output.cols << '\n';
    std::cout << "value(0,0): " << output.at(0, 0) << '\n';
    std::cout << "value(1,3): " << output.at(1, 3) << '\n';
    return 0;
}
