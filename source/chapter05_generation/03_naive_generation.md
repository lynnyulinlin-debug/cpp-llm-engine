# 05.3 Naive Generation 朴素生成

## 学习入口

本节用完整重算建立单请求基线，固定 `forward -> sample -> append -> stop` 的生成闭环。完成后应能指出每一步由哪个对象执行，并解释真实模型 Context 的增量 K/V 优化省掉了什么计算。

- 前置阅读：[`05.2 N 层 Transformer 组装`](./02_transformer_assembly.md)、[`Chapter04 Decoder`](../chapter04_attention/intro.md)
- 源码直达：<a href="../_code/chapter05_generation/src/runtime.cpp"><code>runtime.cpp</code></a>、<a href="../_code/chapter05_generation/src/request.cpp"><code>request.cpp</code></a>、<a href="../_code/chapter05_generation/examples/main_generation.cpp"><code>main_generation.cpp</code></a>
- 相关阅读：[`05.4 KV Cache`](./04_kv_cache.md)、[`05.5 Scheduler`](./05_scheduler.md)、[`05.6 Sampler`](./06_sampler.md)

## 最小生成循环

朴素生成每一步都用当前完整序列执行 forward，再把选出的 token 追加到 Request。下面是概念伪代码；仓库中的真实入口是 `Runtime::generate()` 和 `Runtime::decode_one()`。

```cpp
for (std::size_t step = 0; step < max_new_tokens; ++step) {
    Tensor logits = model.forward(context);
    const int token = sampler.sample(logits);
    request.append_generated(token);
    context.push_back(token);

    if (token == eos_token_id || context.size() >= max_context_length) {
        break;
    }
}
```

停止条件有三个：生成 EOS、达到 `max_new_tokens`、达到模型上下文上限。实现应在下一次 forward 前检查 terminal 状态和 context budget，避免结束后的额外计算。

| 对象 | 在生成闭环中的职责 |
| --- | --- |
| `Request` | 保存 prompt、生成 token、状态和停止原因 |
| `Runtime` | 执行 prefill/decode，追加结果并推进状态 |
| `RequestModelExecutor` | 持有单请求 Model Context，返回 logits |
| `Sampler` | 从 logits 选择下一个 token |
| `Scheduler` | 决定哪个请求在当前 step 被推进 |

## 基线与增量执行

完整重算会重复计算全部历史 token。prompt 长度为 `S`、生成 `T` 个 token 时，Attention 相关工作近似累积为：

```text
S^2 + (S + 1)^2 + ... + (S + T - 1)^2
```

真实 `RealLlamaCpuContext` 在 prefill 后保存每层 K/V，decode 只计算新 token 并读取历史缓存。Chapter05 的教学 `KVCache` 保存 token ID history，用于观察请求状态；它与模型 Context 内的分层 K/V 是不同对象。

朴素路径的价值是提供可读、可重复的正确性基线。引入缓存后仍应使用相同 prompt、Sampler 和停止条件比较输出，避免把算法变化误判为性能收益。

## 验证

```bash
bash code/chapter05_generation/scripts/run.sh
```

检查单请求可以连续生成，EOS 和 token/context budget 都能结束循环，相同输入和 Greedy Sampler 得到固定输出。随后阅读 [`05.4 KV Cache`](./04_kv_cache.md)，分别跟踪 token history、模型分层 K/V 和 block ownership。
