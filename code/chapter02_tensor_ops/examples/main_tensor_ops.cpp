#include "chapter02_tensor_ops/tensor.h"

#include <iostream>

int main() {
    chapter02_tensor_ops::Tensor tensor(2, 3);
    tensor.at(0, 0) = 1.0f;
    tensor.at(0, 1) = 2.0f;
    tensor.at(0, 2) = 3.0f;
    tensor.at(1, 0) = 4.0f;
    tensor.at(1, 1) = 5.0f;
    tensor.at(1, 2) = 6.0f;

    const chapter02_tensor_ops::Tensor reshaped = tensor.reshape(3, 2);
    const chapter02_tensor_ops::Tensor sliced = tensor.slice(0, 2, 1, 2);

    std::cout << "[chapter02] tensor ops demo\n";
    std::cout << "tensor ready: " << std::boolalpha << tensor.ready() << '\n';
    std::cout << "reshaped shape: " << reshaped.rows() << "x" << reshaped.cols() << '\n';
    std::cout << "sliced shape: " << sliced.rows() << "x" << sliced.cols() << '\n';
    std::cout << "shares storage: " << std::boolalpha << tensor.shares_storage_with(reshaped) << '\n';
    std::cout << "slice value(0,0): " << sliced.at(0, 0) << '\n';
    return 0;
}
