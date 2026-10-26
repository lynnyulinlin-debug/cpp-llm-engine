# 05.4 KV Cache：分清历史 token、真实 K/V 与 BlockTable

生成过程需要保存历史状态，但仓库里有四个名称相近、用途不同的对象。阅读本节时先确认讨论的是 token history、K/V 数值，还是物理 block ownership。

## 四种缓存对象

Chapter05 的 `KVCache` 只保存 Request 已消费和生成的 token ID。它让教学 Runtime 可以检查请求历史和生命周期，但不保存 Attention 的 key/value tensor。

| 对象 | 保存内容 | 所有者 | 代码入口 |
| --- | --- | --- | --- |
| Chapter05 `KVCache` | `TokenSequence`，即 token ID 历史 | `Request` | <a href="../_code/chapter05_generation/include/chapter05_generation/kv_cache.h"><code>kv_cache.h</code></a> |
| Chapter04 `KvCache` | 单层连续 `float` key/value，布局为 `[tokens, heads, head_dim]` | Attention 教学测试 | <a href="../_code/chapter04_attention/include/chapter04_attention/kv_cache.h"><code>kv_cache.h</code></a> |
| `RealLlamaCpuContext` | 每层真实 K/V 数值、position 和执行统计 | 单个真实模型请求 | <a href="../_code/chapter01_tensor_loader/src/real_llama_cpu_decoder.h"><code>real_llama_cpu_decoder.h</code></a> |
| `KvBlockPool` / `BlockTable` | 物理 block ID、逻辑 token 到 block/slot 的映射 | Scheduler / ServiceCore | <a href="../_code/chapter05_generation/include/chapter05_generation/serving_preview.h"><code>serving_preview.h</code></a> |

## Chapter05 的实际接口

教学 `KVCache` 的公开操作只有追加 token、清空、读取长度和读取 token 序列。下面代码对应仓库中的真实接口，不代表分层 K/V tensor。

```cpp
KVCache cache;
cache.append(token_id);
const TokenSequence& history = cache.tokens();
cache.clear();
```

`Request` 持有这个对象。Runtime 成功推进后更新 token history；请求结束、取消、超时或失败后，ServiceCore 的统一释放路径会清空它。

## 真实模型怎样保存 K/V

`RealLlamaCpuModel` 只保存共享只读权重。每个请求通过 `create_context()` 获得独立的 `RealLlamaCpuContext`，Context 为每层保存 key/value 数值并维护当前位置。

| 操作 | Context 行为 | logits 行为 |
| --- | --- | --- |
| 首个 prefill chunk | `reset()` 后逐 token 写入各层 K/V | 只有 final chunk 请求最后一个 token 的 logits |
| 后续 prefill chunk | 在已有 position 和分层 K/V 后追加 | 中间 chunk 不产生 logits |
| decode | 读取历史 K/V，追加一个 token 的 K/V | 产生下一次采样所需 logits |
| reset / 请求释放 | position 归零并清空每层 key/value | 不保留上一个请求状态 |

## BlockTable 管理什么

`KvBlockPool` 根据 token 数预留物理 block ID，`BlockTable` 把逻辑 token 位置映射成 `{physical_block, slot}`。当前对象只管理容量和 ownership，不存储 `RealLlamaCpuContext` 中的 K/V 数值。

| 操作 | 作用 | 失败时要求 |
| --- | --- | --- |
| `reserve(request_id, total_tokens)` | 为目标 token 数预留足够 block | 容量不足时不做部分修改 |
| `commit_tokens(...)` | 提交已经成功执行的 token 数 | 不能超过已预留容量 |
| `trim_to_tokens(...)` | executor 未推进时撤销多余预留 | 保留已提交 token 对应 block |
| `locate(...)` | 返回逻辑 token 的物理 block 和 slot | 越界必须明确失败 |
| `release(request_id)` | 请求终态时归还全部 block | 不影响其他请求的 block table |

## Prefill 与 Decode 的状态变化

Runtime 同时更新 Request 可见状态和真实 Context，但两者承担不同职责。Scheduler 只决定本 step 推进哪些 token，不直接读写 K/V 数值。

| 阶段 | Request token history | 真实 Model Context | BlockTable |
| --- | --- | --- | --- |
| Prefill 开始 | 保存已提交 prompt token | 建立各层 K/V 和 position | 预留并提交 prompt token 容量 |
| Chunked prefill | `prefill_offset` 逐段前进 | 在原 Context 后追加 K/V | 每段成功后提交 token count |
| Decode | 追加采样得到的 token | 用历史 K/V 计算并追加新 K/V | 为新 token 提交位置 |
| 终态释放 | `clear()` | Executor/Context 析构或 reset | `release()` 归还物理 block |

## 阅读和验证

按“请求状态 → 单层 K/V 数值 → 真实分层 Context → block ownership”的顺序阅读，避免从 `KVCache` 名称推断对象内容。

| 验证对象 | 测试入口 | 通过标准 |
| --- | --- | --- |
| Chapter05 token history | <a href="../_code/chapter05_generation/tests/kv_cache_test.cpp"><code>kv_cache_test.cpp</code></a> | append、size、tokens 和 clear 正确 |
| Chapter04 连续 K/V | <a href="../_code/chapter04_attention/tests/chapter04_attention_test.cpp"><code>chapter04_attention_test.cpp</code></a> | append、容量拒绝、decode offset 和 clear 正确 |
| `KvBlockPool` | <a href="../_code/chapter05_generation/tests/serving_preview_test.cpp"><code>serving_preview_test.cpp</code></a> | reserve、commit、rollback、locate 和 release 正确 |
| 真实分层 K/V | `real_tinyllama_generation_test`、`real_tinyllama_serving_test` | golden token、chunked prefill、Context 隔离和终态释放同时通过 |
