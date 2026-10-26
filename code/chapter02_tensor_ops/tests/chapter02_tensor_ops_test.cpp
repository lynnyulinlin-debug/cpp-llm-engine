// Chapter 02：Tensor view 综合测试
// 职责：验证 reshape、slice、offset、stride、别名写入、边界和生命周期。
// 通过条件：CTest 中唯一测试通过，父 Tensor 释放后 view 仍可访问。
// 验证：bash code/chapter02_tensor_ops/scripts/run.sh。
#include "chapter02_tensor_ops/tensor.h"

#include <cassert>
#include <stdexcept>

namespace {

template <typename Function>
void expect_out_of_range(Function&& function) {
    bool threw = false;
    try {
        function();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

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
    const auto tensor_contract = tensor.contract();
    assert(tensor_contract.shape.dimensions == std::vector<std::size_t>({2, 3}));
    assert(tensor_contract.strides == std::vector<std::size_t>({3, 1}));
    assert(tensor_contract.storage_offset_elements == 0);
    assert(tensor_contract.validate_element_view(sizeof(float)).is_ok());

    const chapter02_tensor_ops::Tensor reshaped = tensor.reshape(3, 2);
    assert(reshaped.ready());
    assert(reshaped.rows() == 3);
    assert(reshaped.cols() == 2);
    assert(reshaped.offset() == 0);
    assert(reshaped.row_stride() == 2);
    assert(reshaped.col_stride() == 1);
    assert(reshaped.shares_storage_with(tensor));
    assert(reshaped.at(0, 0) == 1.0f);
    assert(reshaped.at(0, 1) == 2.0f);
    assert(reshaped.at(1, 0) == 3.0f);
    assert(reshaped.at(2, 1) == 6.0f);

    chapter02_tensor_ops::Tensor sliced = tensor.slice(0, 2, 1, 2);
    assert(sliced.ready());
    assert(sliced.rows() == 2);
    assert(sliced.cols() == 2);
    assert(sliced.offset() == 1);
    assert(sliced.row_stride() == 3);
    assert(sliced.col_stride() == 1);
    assert(sliced.shares_storage_with(tensor));
    const auto sliced_contract = sliced.contract();
    assert(sliced_contract.shape.dimensions == std::vector<std::size_t>({2, 2}));
    assert(sliced_contract.strides == std::vector<std::size_t>({3, 1}));
    assert(sliced_contract.storage_offset_elements == 1);
    assert(sliced_contract.validate_element_view(sizeof(float)).is_ok());
    assert(sliced.at(0, 0) == 2.0f);
    assert(sliced.at(1, 1) == 6.0f);

    sliced.at(0, 0) = 42.0f;
    assert(tensor.at(0, 1) == 42.0f);

    const chapter02_tensor_ops::Tensor contiguous_offset_view = tensor.slice(1, 1, 0, 3);
    assert(contiguous_offset_view.contiguous());
    const chapter02_tensor_ops::Tensor offset_reshaped = contiguous_offset_view.reshape(3, 1);
    assert(offset_reshaped.ready());
    assert(offset_reshaped.offset() == 3);
    assert(offset_reshaped.at(0, 0) == 4.0f);
    assert(offset_reshaped.at(2, 0) == 6.0f);

    const chapter02_tensor_ops::Tensor invalid = sliced.reshape(4, 1);
    assert(!invalid.ready());

    chapter02_tensor_ops::Tensor explicit_output;
    const auto non_contiguous_status = sliced.try_reshape(4, 1, &explicit_output);
    assert(!non_contiguous_status.is_ok());
    assert(non_contiguous_status.message() == "reshape requires a contiguous tensor");
    const auto shape_status = tensor.try_reshape(4, 2, &explicit_output);
    assert(!shape_status.is_ok());
    assert(shape_status.message() == "reshape element count must not change");
    assert(!tensor.try_reshape(3, 2, nullptr).is_ok());

    assert(!tensor.slice(2, 1, 0, 1).ready());
    assert(!tensor.slice(0, 1, 3, 1).ready());
    assert(!tensor.slice(0, 0, 0, 1).ready());
    const auto slice_status = tensor.try_slice(1, 2, 0, 1, &explicit_output);
    assert(!slice_status.is_ok());
    assert(slice_status.message() == "slice range exceeds tensor shape");
    assert(!tensor.try_slice(0, 1, 0, 1, nullptr).is_ok());
    expect_out_of_range([&] { tensor.at(2, 0); });
    expect_out_of_range([&] { tensor.at(0, 3); });

    chapter02_tensor_ops::Tensor retained_view;
    {
        chapter02_tensor_ops::Tensor owner(1, 2);
        owner.at(0, 0) = 7.0f;
        retained_view = owner.slice(0, 1, 0, 2);
        assert(owner.storage_use_count() == 2);
        assert(retained_view.storage_use_count() == 2);
    }
    assert(retained_view.ready());
    assert(retained_view.at(0, 0) == 7.0f);
    assert(retained_view.storage_use_count() == 1);

    return 0;
}
