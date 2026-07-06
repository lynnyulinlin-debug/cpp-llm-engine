# 05.4 KV Cache 缓存


## 1. 这一页要解决什么

这一页要把 KV Cache 设计成生成 runtime 里的显式状态，而不是一段藏在前向函数内部的临时逻辑。  
它的直接目标是减少增量推理中的重复计算。

## 阅读入口

- 前置阅读：[`05.3 Naive Generation`](/chapter05_generation/03_naive_generation)、[`05.2 N 层 Transformer 组装`](/chapter05_generation/02_transformer_assembly)
- 源码直达：[`chapter05_generation 代码入口`](/code/chapter05_generation)
- 相关阅读：[`05.5 Sampler`](/chapter05_generation/05_sampler)、[`05.6 端到端生成测试`](/chapter05_generation/06_end_to_end_generation)

## 2. 为什么这一步重要

KV Cache 是生成性能里最关键的优化点之一。  
它让 decode 阶段不必每次都从头计算完整序列，而是只处理新增 token。

更重要的是，它把“生成 runtime 的状态”变成了可显式管理的对象。  
没有 cache，decode 每一步都会重复处理历史上下文；有了 cache，历史 K/V 就变成了可以复用的运行时资产。

## 3. 最小设计长什么样

最小 cache 只需要把两件事说清楚：

- 每层缓存历史 K/V
- 新 token 只追加到缓存

最小结构可以写成：

```cpp
struct KVCache {
    std::vector<Tensor> key_per_layer;
    std::vector<Tensor> value_per_layer;
};
```

然后每次 decode 时，只追加新 token 对应的 K/V。  
源码锚点：`code/chapter05_generation/src/kv_cache.cpp`

这不是为了省一个循环，而是为了把“复用历史状态”这件事变成显式行为。

### cache 的维度布局

KV Cache 的张量布局通常有两种常见写法：

- `[S, H, D]`：序列维度在前，便于按 token 追加
- `[H, S, D]`：head 维度在前，便于和某些 attention kernel 对齐

本教程建议先采用 `[S, H, D]`，原因是：

- `push_back` 语义清晰
- 按 token 位置索引直观
- 更容易和“追加缓存”这个概念对应起来

### prefill 和 decode 的 cache 契约

| 阶段 | Cache 状态 | 输入 | 输出 |
| --- | --- | --- | --- |
| Prefill | 空 -> 完整 | 完整 prompt `[1, S]` | 填充 cache + logits |
| Decode | 完整 -> 追加 | 新 token `[1, 1]` | 追加 cache + logits |

关键区别是：

- Prefill 一次性计算完整 prompt 的 K/V
- Decode 每一步只计算一个新 token 的 K/V

对应接口可以理解为：

```cpp
Tensor prefill(const std::vector<int>& prompt, KVCache& cache);
Tensor decode(int token, KVCache& cache);
```

源码锚点：`code/chapter05_generation/src/kv_cache.cpp`

## 4. 为什么 cache 不能和全局绑定

从工程角度倒推，KV Cache 必须和请求绑定，而不是和全局绑定，原因很直接：

- 每个请求的上下文不同
- 每个请求的 decode 长度不同
- 每个请求的结束时机不同
- 多请求并发时，cache 必须能区分归属

所以这一页真正要建立的不是“一个 cache 容器”，而是“cache 如何和请求、层、token 生命周期对齐”的设计约束。  
如果把 cache 设计成全局共享但没有请求隔离，单请求时可能看不出问题，但一旦进入多请求场景，历史状态就会互相污染。

### 请求隔离怎么做

每个生成请求都应该拥有自己独立的 cache：

```cpp
class GenerationRequest {
    std::vector<int> prompt_;
    KVCache cache_;
    std::vector<int> generated_;
    bool finished_;
};
```

这样 scheduler 在推进某个请求时，只会操作它自己的 cache，不会把别的请求的历史状态混进来。

### cache 的内存管理

每个请求的 KVCache 在长上下文场景里会占用不少内存，所以工程上通常要考虑：

1. 按需分配，只分配当前需要的 `max_seq_len`
2. 内存池复用，减少反复申请和释放
3. 量化存储，后续可以考虑用更小精度存 K/V

从内存视角看，cache 的生命周期应该和请求生命周期绑定：

- 请求开始时分配
- 请求进行时追加
- 请求结束时释放或回收

## 5. 怎么验证

- 历史 K/V 能正确保存
- 新 token 只处理增量部分
- 与 naive generation 结果一致或接近

这里“接近”是指在相同输入和相同采样策略下，KV Cache 版和 naive 版应该保持一致的生成逻辑，只是前者少算历史部分。

## 6. 常见问题

- cache 维度和层数对不上
- 历史状态覆盖错误
- 缓存逻辑和模型前向耦合过深
- request 隔离不清，导致多请求场景互相污染
- prefill 和 decode 使用了不同的 cache contract
- cache 生命周期和请求生命周期没有绑在一起

## 7. 和后续章节怎么接

这一页把 decoder 的历史状态显式化，下一页才能继续把这些状态接到 sampler 和端到端测试上。  
 它不是附属优化，而是 generation runtime 从“能跑”走向“可持续运行”的关键台阶。
