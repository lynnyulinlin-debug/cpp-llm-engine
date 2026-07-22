# 05.3 Naive Generation 朴素生成


## 1. 这一页要解决什么

这一页要把“能生成”先做出来，而且是用最朴素、最容易理解的方式做出来。先不要 KV Cache、复杂调度或流式输出，只先确认生成闭环能跑通。

## 阅读入口

- 前置阅读：[`05.2 N 层 Transformer 组装`](/chapter05_generation/02_transformer_assembly)、[`chapter04_attention`](/chapter04_attention/intro)
- 源码直达：[`chapter05_generation 代码入口`](/code/chapter05_generation)
- 相关阅读：[`05.4 KV Cache`](/chapter05_generation/04_kv_cache)、[`05.5 Scheduler`](/chapter05_generation/05_scheduler)、[`05.6 Sampler`](/chapter05_generation/05_sampler)

## 代码直达

- [`src/runtime.cpp`](/code/chapter05_generation#runtime-cpp)
- [`src/request.cpp`](/code/chapter05_generation#request-cpp)
- [`src/sampler.cpp`](/code/chapter05_generation#sampler-cpp)
- [`src/scheduler.cpp`](/code/chapter05_generation#scheduler-cpp)
- [`examples/main_generation.cpp`](/code/chapter05_generation#main-generation-cpp)
- [`tests/runtime_smoke_test.cpp`](/code/chapter05_generation#runtime-smoke-test-cpp)

## 2. 为什么要先做朴素生成

优化必须有基线。朴素生成虽然慢，但它把生成流程完整展开了：prompt 怎么进入、logits 怎么出来、token 怎么回写、什么时候停止，都能被看见。

如果没有这个基线，后面所有 cache、scheduler 和 streaming 的改动都很难判断到底是收益、代价还是噪声。

## 3. 这条基线长什么样

最小实现只保留三个原则：

- 每次生成都重新计算完整序列
- 不缓存历史 K/V
- 用最简单的循环确认链路正确

这个版本可以慢，但不能乱。只要它能稳定地产生 token，后面的 KV Cache 优化就有了比较对象。

### 生成停止条件

生成循环通常在以下任一条件满足时停止：

1. 达到最大长度：`step >= max_new_tokens`
2. 生成 EOS token：`next_token == eos_token_id`
3. 达到上下文限制：`context.size() >= max_context_length`

最小伪代码可以写成：

```cpp
while (step < max_new_tokens && next_token != eos_token_id) {
    // 生成逻辑
}
```

实际实现里，通常先检查 EOS，再检查长度限制，避免在已经结束时继续做无意义计算。

### Sampler 的职责

Sampler 负责把模型输出的 logits 变成下一个 token：

- 取最后一个位置的 logits
- 转成概率分布
- 按策略选 token

常见策略包括：

- greedy：选概率最大的 token
- top-k：只在 top-k 里采样
- top-p：在累积概率达到阈值的集合里采样

朴素生成里，最简单的起点就是 greedy sampler。后面如果要扩展 sampling 策略，可以在不改生成主循环的前提下替换 sampler。

## 4. 为什么这一步能讲清楚生成

生成系统最容易被忽略的，不是模型，而是状态流转。  
朴素生成把状态流转暴露出来以后，读者就能看见生成循环如何运转：

- 上下文是怎么进入的
- logits 是怎么出来的
- token 是怎么回写的
- 什么时候应该停止

最小伪代码可以写成：

```cpp
for (int step = 0; step < max_new_tokens; ++step) {
    Tensor logits = model.forward(context);
    int next_token = sampler.sample(logits);
    context.push_back(next_token);
}
```

这段逻辑的价值不在于快，而在于它最容易把生成闭环暴露出来。  
有了它，后面再引入 KV Cache，读者才知道优化到底优化了哪一步。

### 完整重算的计算代价

朴素生成每一步都会重新计算完整上下文，所以计算量会随着序列长度不断增长。

假设：

- 序列长度 `S = 100`
- 新生成 token 数 `T = 50`

那么 naive generation 会做类似下面的重复计算：

```text
100^2 + 101^2 + 102^2 + ... + 149^2
```

而 KV Cache 版本只需要对新 token 做增量计算，历史部分不再重复算。  
当序列更长时，这个差异会迅速放大，所以 naive generation 的作用是提供基线，不是提供性能。

### 这一层和 runtime 状态怎么接

朴素生成不是一个孤立循环，它实际上是 runtime 的最小闭环：

1. `request` 提供 prompt 和当前生成状态
2. `runtime` 读取当前上下文，跑一次 forward
3. `sampler` 把 logits 变成下一步 token
4. `request` 把 token 追加回自己的上下文
5. `scheduler` 决定下一个该推进哪个请求

最小交互可以写成：

```cpp
while (!request.finished()) {
    Tensor logits = runtime.forward(request.context());
    int token = sampler.sample(logits);
    request.append(token);
    scheduler.complete_or_requeue(request);
}
```

这一步真正要讲清楚的是：生成循环围绕 `request` 进行，`sampler` 只负责选 token，`scheduler` 只负责决定推进顺序，`runtime` 只负责一次 forward。

这样后面引入 KV Cache 时，读者就能清楚知道“优化的是哪一段”，而不是把生成看成一个黑盒 while loop。

## 5. 怎么验证

- 可以生成连续 token
- 输出流程正确
- 行为符合“不缓存”的预期

如果在相同 prompt 下，naive generation 每次都会重新走完整前向，但输出过程仍然稳定，那说明基线已经可用了。

## 6. 常见问题

- 生成逻辑和 forward 逻辑混淆
- 输入序列重复计算却没有记录出来
- 基线和优化版本混在一起比较

## 7. 和后续章节怎么接

这一页把生成循环先跑通，下一页才能引入 KV Cache。  
KV Cache 要解决的具体问题是：在朴素生成里，每一步都会重新计算历史 token 的 K/V；而在缓存版里，历史 K/V 可以保留，只对新 token 做增量计算。

只要 baseline 立住，后面的优化就有了可比较的起点。
