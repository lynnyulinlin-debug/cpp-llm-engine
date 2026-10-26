# Core Contracts 核心契约

`code/core` 提供 CPU reference 和现有 CUDA backend 共用的最小类型契约，不实现具体 Tensor storage，也不依赖第三方库。

## 资产总览

| 资产 | 作用 |
| --- | --- |
| `include/` | Tensor view、`Status`、Backend、Operator 与 Decoder 公共契约 |
| `tests/` | shape、storage、接口身份和 Decoder contract 校验 |
| `CMakeLists.txt` | 独立 library 和 CTest 构建 |

## 当前边界

- `TensorContract` 描述 shape、stride、storage offset、dtype、device 和 storage bytes，不拥有 storage。
- `quantization_contract.h` 统一 Q4_K/Q6_K/Q8_0 block 尺寸和矩阵 payload 字节数校验。
- `Shape` 负责带溢出检查的元素数量计算。
- `validate_element_view()` 检查普通元素 view 的最大可达位置；Q4_K/Q6_K/Q8_0 packed payload 继续由量化 backend 按 block 校验。
- `Status` 在 backend 边界传递显式成功或失败。
- `Backend` 和 `Operator` 只定义分派身份，不选择 CPU 或 CUDA。
- `DecoderLayerContract` 描述层数、维度和结构特征；`DecoderExecutionContext` 描述一次请求的 `reset/prefill/decode` 执行能力。
- 公共契约不加载 GGUF、不实现 Decoder 数学，也不管理 Scheduler。

## 运行与验证

```bash
cmake -S code/core -B /tmp/cpp-llm-engine-core-build
cmake --build /tmp/cpp-llm-engine-core-build
ctest --test-dir /tmp/cpp-llm-engine-core-build --output-on-failure
```
