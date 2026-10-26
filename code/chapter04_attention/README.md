# Chapter 04：Attention 与 Decoder 结构

本目录把 MatMul、Attention、LayerNorm、FFN 组合成可测试的 DecoderLayer，为 Chapter 05 generation runtime 提供结构输入。完整学习说明见 [`source/chapter04_attention`](../../source/chapter04_attention/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `include/` | Attention、DecoderLayer、backend 和结构契约适配 |
| `src/attention.cpp` | causal multi-head attention |
| `src/attention_backend.cpp` | CPU reference attention backend 适配 |
| `src/attention_kernel.cpp` | CPU causal attention kernel |
| `src/kv_cache.cpp` | 固定容量 KV Cache append/clear |
| `src/layer_norm.cpp` | LayerNorm |
| `src/feed_forward.cpp` | FFN 和激活 |
| `src/decoder_layer.cpp` | Attention、残差和 FFN 组合，并输出 `DecoderLayerContract` |
| `src/linear_layer.cpp` | Decoder 内部线性层 |
| `tests/chapter04_attention_test.cpp` | Attention、decode offset、KV Cache 和输出测试 |
| `examples/main_attention.cpp` | Decoder 结构演示 |
| `scripts/run.sh` | 隔离构建、CTest 和 demo 入口 |
| `code/core` | 共享 backend 和 operator 契约 |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter04_attention/scripts/run.sh
```

本章验证单层结构、教学 KV Cache 和 CPU 结果，不承担请求级 KV 生命周期、完整模型装配或多请求调度。
