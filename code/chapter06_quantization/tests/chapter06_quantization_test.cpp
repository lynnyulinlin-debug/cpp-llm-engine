#include "chapter06_quantization/dequantize_op.h"
#include "chapter06_quantization/linear_layer.h"
#include "chapter06_quantization/quantized_linear.h"

#include <cassert>
#include <cmath>

int main() {
    using namespace chapter06_quantization;

    const Tensor quantized = Tensor::from_int8(2, 2, {1, 2, 3, 4}, 0.1f);
    assert(quantized.ready());
    assert(quantized.is_int8());
    assert(std::abs(quantized.at(0, 0) - 0.1f) < 1e-5f);
    assert(std::abs(quantized.at(1, 1) - 0.4f) < 1e-5f);

    const DequantizeOp dequantize_op;
    const Tensor dequantized = dequantize_op.forward(quantized);
    assert(dequantized.ready());
    assert(dequantized.is_float());
    assert(std::abs(dequantized.at(0, 0) - 0.1f) < 1e-5f);
    assert(std::abs(dequantized.at(1, 1) - 0.4f) < 1e-5f);

    const Tensor input = Tensor::from_float(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    const LinearLayer reference_linear(2, 2);
    const QuantizedLinear quantized_linear(2, 2, 0.1f);

    const Tensor reference_output = reference_linear.forward(input);
    const Tensor quantized_output = quantized_linear.forward(input);

    assert(reference_output.ready());
    assert(quantized_output.ready());
    assert(reference_output.rows() == quantized_output.rows());
    assert(reference_output.cols() == quantized_output.cols());
    assert(std::abs(reference_output.at(0, 0) - quantized_output.at(0, 0)) < 1e-5f);
    assert(std::abs(reference_output.at(1, 1) - quantized_output.at(1, 1)) < 1e-5f);

    return 0;
}
