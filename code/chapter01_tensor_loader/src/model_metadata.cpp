// Chapter 01：模型 metadata 对象实现
// 职责：提供架构、维度、层数和 special token metadata 的有效性检查。
// 契约：valid() 只检查对象字段，不访问外部文件。
// 验证：chapter01_tensor_loader_test.cpp。
#include "model_metadata.h"

namespace chapter01 {

bool ModelMetadata::valid() const {
    return !architecture.empty() && vocab_size > 1 && hidden_size > 0 &&
           max_position > 0 && layer_count > 0;
}

}  // namespace chapter01
