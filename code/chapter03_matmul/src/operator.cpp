// Chapter 03：Operator 接口编译单元
// 职责：为纯接口 Operator 提供独立编译入口，不承载具体计算。
// 契约：具体算子在子类中实现 compute 和 backend 适配。
// 验证：operator_smoke_test.cpp。
#include "chapter03_matmul/operator.h"

namespace chapter03_matmul {

// Intentionally empty: Operator is a pure interface boundary for chapter03.

}  // namespace chapter03_matmul
