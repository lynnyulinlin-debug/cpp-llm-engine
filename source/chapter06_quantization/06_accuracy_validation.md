# 06.6 Accuracy Validation 精度验证


## 用 FP 基线检查误差

精度验证要回答一个很关键的问题：量化模型和 FP16 模型之间的差异，是否还在可接受范围内？

量化的价值不是“更小”本身，而是“在可接受精度下更高效”。  
没有精度验证，就无法判断当前方案是否真的适合部署。

**阅读入口**

- 前置阅读：[`06.5 Quantized Linear`](./05_linear_with_quant.md)、[`06.4 DequantizeOp`](./04_dequantize_op.md)
- 源码直达：<a href="../_code/chapter06_quantization/tests/chapter06_quantization_test.cpp"><code>chapter06_quantization_test.cpp</code></a>、<a href="../_code/chapter06_quantization/src/linear_layer.cpp"><code>linear_layer.cpp</code></a>、<a href="../_code/chapter06_quantization/src/quantized_linear.cpp"><code>quantized_linear.cpp</code></a>
- 相关阅读：[`章节导读`](./intro.md)、[`chapter07_benchmark`](../chapter07_benchmark/intro.md)

## 验证设计

精度验证的核心，不是简单跑一个 pass/fail，而是把“量化节省了什么、牺牲了什么”清楚表达出来。  
对真实部署来说，能接受的误差阈值往往和任务、模型、硬件有关，所以精度验证的重点是建立方法，而不是写死一个数字。

建议把验证结果拆成两层：

- 一层看局部 logits 或中间 tensor
- 一层看最终输出

这样既能判断量化策略是否整体可用，也能在失败时快速定位是哪一层开始偏离。

局部验证的锚点通常选择在量化路径与 FP16 路径共享接口的位置，比如 LayerNorm 输出、Attention 输出或 FFN 输出。在这些位置插入比较点，可以判断误差是在特定模块内部积累，还是在跨模块传播中被放大。

对照方式：

最小对照伪代码可以写成：

```python
golden = fp16_model(x)
quant = quant_model(x)
score = cosine_similarity(golden, quant)
```

这足以表达“比较什么、怎么比较”。

## 验证与排查

本章的 Q4_K CPU reference 已使用真实 TinyLlama 的 `token_embd.weight` 与 `llama.cpp` 官方 `dequantize_row_q4_K` 做交叉验证：共比较 65,536,000 个元素，`max_abs_error=0`、`max_rel_error=0`。Linear 测试还使用非均匀 scale/min/quant 和非均匀输入，对照 reference、scalar fused 与 AVX2/FMA fused 的输出。

Q6_K CPU reference 也已使用真实 TinyLlama 的 `blk.0.ffn_down.weight` 与 `llama.cpp` 官方 `dequantize_row_q6_K` 做交叉验证：共比较 11,534,336 个元素，payload 为 9,461,760 字节，`max_abs_error=0`、`max_rel_error=0`。Q6_K Linear 已有 reference、scalar fused 和 AVX2/FMA fused 路径，并已进入真实 TinyLlama 22 层 Decoder。

Q8_0 使用 32 个值、34 字节的 block。合成测试对照完整解码 reference、直接 decode-and-dot fused backend 和统一 policy；真实 SmolVLM 路径覆盖 Q8_0 token embedding、Attention、FFN 与 LM Head。相同 token 分别使用 token-prefill 和预计算 embedding-prefill 时 logits 完全一致；最小图文 `3x960` 输入经过完整 32 层后，C++/NumPy top-1 都是 token `2`，top-1 logit 分别为 `11.18585682` 和 `11.1858902`。

AVX2 单变量实验先用合成 block 对照 reference，再用真实 Decoder 的固定 golden token、顶层计时闭合和 FFN 子项闭合作为闸门。Q6_K down 的三轮归档报告记录 FFN down 从 `1355.83 ms` 降到 `792.08 ms`，Decoder 总耗时从 `5104.69 ms` 降到 `4433.33 ms`。早期单变量文本没有内嵌 CPU 和编译器字段，因此只能作为仓库历史测量，不能用于跨机器性能结论；原始数据见 <a href="../_code/chapter08_profiling/results/tinyllama_q6k_down_ab.txt"><code>tinyllama_q6k_down_ab.txt</code></a>。

Q4_K 实验保持 down 与其他 projection 为 scalar，只切换 FFN gate/up。三轮归档中，gate 从 `1243.69 ms` 降到 `212.87 ms`，up 从 `1242.09 ms` 降到 `213.71 ms`，Decoder 总耗时从 `5004.27 ms` 降到 `2970.35 ms`，判定 `KEEP`。Q4_K 与 Q6_K 各有独立的配对 scalar baseline，不能跨实验直接相减绝对时间；原始数据见 <a href="../_code/chapter08_profiling/results/tinyllama_q4k_ffn_ab.txt"><code>tinyllama_q4k_ffn_ab.txt</code></a>。

Attention 实验使用独立 policy，只切换 Q、K、V 和 output projection，FFN 与 LM Head 保持 scalar。三轮归档中，QKV 从 `587.50 ms` 降到 `124.02 ms`，Attention output 从 `466.30 ms` 降到 `78.83 ms`，Decoder 总耗时从 `5161.73 ms` 降到 `4266.20 ms`，判定 `KEEP`；原始数据见 <a href="../_code/chapter08_profiling/results/tinyllama_q4k_attention_ab.txt"><code>tinyllama_q4k_attention_ab.txt</code></a>。

组合实验随后同时启用 Q4_K Attention、Q4_K FFN 和 Q6_K FFN AVX2 backend，LM Head 保持 scalar。Decoder 总耗时从 `5208.81 ms` 降到 `1078.23 ms`，双请求 ServiceCore dispatch 从 `7573.39 ms` 降到 `1629.51 ms`，吞吐从 `1.0564` 提升到 `4.9095 token/s`；golden、Context/cache 隔离和 terminal release 均通过。该归档明确记录 AMD Ryzen 9 9955HX、GCC 13.1.0、Release、warmup 1 次和测量 3 次；详见 <a href="../_code/chapter08_profiling/results/tinyllama_optimized_service_scalar.json"><code>scalar JSON</code></a>、<a href="../_code/chapter08_profiling/results/tinyllama_optimized_service_avx2.json"><code>AVX2 JSON</code></a> 和 <a href="../_code/chapter08_profiling/results/tinyllama_optimized_policy_ab.txt"><code>A/B 报告</code></a>。该模式需要显式选择，正常生成和服务继续默认 scalar。

跨章集成 benchmark 还会把同一个真实 GGUF Tensor 交给 Chapter 06 的两条 Linear 路径：

```bash
cmake -S code/chapter01_tensor_loader \
  -B /tmp/chapter01-q4k-build \
  -DCHAPTER06_SOURCE_DIR="$PWD/code/chapter06_quantization"
cmake --build /tmp/chapter01-q4k-build
/tmp/chapter01-q4k-build/gguf_q4k_benchmark \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --warmup 1 --iterations 3 --max-tensors 3
```

输出契约如下：

| 字段 | 含义 |
| --- | --- |
| `min/mean/p50/max` | 每个 tensor、每条路径的多轮耗时统计，单位为 ms |
| `max_abs_delta` | reference 与 fused 输出的最大绝对误差 |
| `shape/payload_bytes` | 本次真实权重的输入规模 |

`gguf_q4k_benchmark` 的示例数字没有保存为带环境字段的基线文件，因此命令只作为复现实验入口，不把某次本地耗时作为教程结论。可归档的性能判断使用 Chapter08 的结果格式；量化正确性继续由 Chapter06 自动测试中的 reference/fused 差值约束负责。

验证时重点看：

- 使用相同输入
- 比较输出向量或 logits
- 观察是否达到阈值

如果能在多个输入样本上都保持相近的误差区间，就说明这条量化路径的稳定性更可信。

常见问题：

- 只看单个样本
- 指标阈值设置不合理
- 没有区分局部误差和整体误差
