// Chapter 02：Tensor view 示例
// 职责：演示 reshape、slice、offset/stride 和共享 storage 的可观察结果。
// 运行：bash code/chapter02_tensor_ops/scripts/run.sh。
// 验证：输出 shares storage=true，并打印 slice offset/stride。
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
    std::cout << "slice offset/stride: " << sliced.offset() << "/" << sliced.row_stride() << "/"
              << sliced.col_stride() << '\n';
    std::cout << "shares storage: " << std::boolalpha << tensor.shares_storage_with(reshaped) << '\n';
    std::cout << "slice value(0,0): " << sliced.at(0, 0) << '\n';
    return 0;
}
