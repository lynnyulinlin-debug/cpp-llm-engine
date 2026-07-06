#include "chapter02_tensor_ops/tensor.h"

#include <cassert>

int main() {
    chapter02_tensor_ops::Tensor tensor(2, 3);
    tensor.at(0, 0) = 1.0f;
    tensor.at(0, 1) = 2.0f;
    tensor.at(0, 2) = 3.0f;
    tensor.at(1, 0) = 4.0f;
    tensor.at(1, 1) = 5.0f;
    tensor.at(1, 2) = 6.0f;

    assert(tensor.ready());
    assert(tensor.contiguous());

    const chapter02_tensor_ops::Tensor reshaped = tensor.reshape(3, 2);
    assert(reshaped.ready());
    assert(reshaped.rows() == 3);
    assert(reshaped.cols() == 2);
    assert(reshaped.shares_storage_with(tensor));
    assert(reshaped.at(0, 0) == 1.0f);
    assert(reshaped.at(0, 1) == 2.0f);
    assert(reshaped.at(1, 0) == 3.0f);
    assert(reshaped.at(2, 1) == 6.0f);

    chapter02_tensor_ops::Tensor sliced = tensor.slice(0, 2, 1, 2);
    assert(sliced.ready());
    assert(sliced.rows() == 2);
    assert(sliced.cols() == 2);
    assert(sliced.shares_storage_with(tensor));
    assert(sliced.at(0, 0) == 2.0f);
    assert(sliced.at(1, 1) == 6.0f);

    sliced.at(0, 0) = 42.0f;
    assert(tensor.at(0, 1) == 42.0f);

    const chapter02_tensor_ops::Tensor invalid = sliced.reshape(4, 1);
    assert(!invalid.ready());
    return 0;
}
