// Chapter 04：DecoderLayer 示例
// 职责：构造 2x4 输入，运行单层 Decoder 并打印输出 shape 和数值。
// 运行：bash code/chapter04_attention/scripts/run.sh。
// 验证：输出 ready=true 和 2x4 结果。
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
