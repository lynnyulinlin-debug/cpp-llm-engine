# 02.5 Tensor 与 View 框架对照：数据不动，观察方式改变

本章的 `reshape` 和 `slice` 共享一份 CPU storage。对照成熟实现时，先检查 shape、stride、offset 和 storage，再判断操作是否真的零拷贝。

## 职责映射

| 教程对象 | ggml / `llama.cpp` | vLLM runtime | 阅读重点 |
| --- | --- | --- | --- |
| `TensorStorage` | tensor data buffer / backend buffer | cache tensor 和执行 buffer | 谁拥有底层内存 |
| `shape`、`stride`、`offset` | `ggml_tensor` 元数据 | block table、slot mapping、tensor view | 逻辑索引怎样落到物理位置 |
| `reshape` | reshape/view graph op | model runner 中的 shape 变换 | 是否要求连续布局 |
| `slice` | view/offset 操作 | KV block 或 token window 寻址 | 子窗口是否复制数据 |
| `TensorContract` | backend 的 shape/type 检查 | worker/backend 输入契约 | 错误在哪一层被拒绝 |

## 源码入口

ggml 适合观察通用 tensor/view，vLLM 适合观察这些语义怎样进入分页 KV Cache。

| 框架 | 官方入口 | 带着什么问题读 |
| --- | --- | --- |
| ggml | [`ggml.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml.c) | tensor 元数据和 view offset 怎样表达 |
| ggml | [`ggml-backend.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp) | storage 进入不同 backend 后由谁管理 |
| vLLM | [`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) | 逻辑 token 怎样映射到可复用 cache block |

## 教程保留与省略

本章限制在连续 CPU storage，便于用指针、offset 和测试直接证明共享关系。

| 已实现 | 暂不实现 | 省略原因 |
| --- | --- | --- |
| 共享 storage 的 reshape 和 slice | transpose、广播、任意非连续 view | 先固定最小地址计算 |
| shape/stride/offset contract | 自动 contiguous copy | 不隐藏潜在复制成本 |
| CPU 生命周期与越界检查 | GPU allocator、stream 和异步释放 | 设备生命周期留给 backend 主线 |
| 线性窗口 slice | 分页 KV、prefix sharing 和 copy-on-write | cache 策略由 Chapter05/09 承接 |

## 源码阅读问题

1. 一个 view 销毁后，底层 storage 为什么仍可能有效？
2. reshape 在什么条件下可以只改元数据，什么时候必须复制？
3. 分页 KV Cache 的 block table 与本章的 offset/stride 有什么共同点？
