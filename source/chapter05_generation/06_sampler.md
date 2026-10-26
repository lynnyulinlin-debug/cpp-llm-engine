# 05.6 Sampler 采样器

## 把 Logits 变成 Token

先用 Greedy 把模型输出的 logits 转成 token，再理解 Temperature、Top-K 和 Top-p 的扩展位置。采样是生成链路的最后一项决策，不改变模型前向。

**阅读入口**

- 前置阅读：[`05.5 Scheduler`](./05_scheduler.md)、[`05.4 KV Cache`](./04_kv_cache.md)、[`05.3 Naive Generation`](./03_naive_generation.md)
- 源码直达：<a href="../_code/chapter05_generation/src/sampler.cpp"><code>sampler.cpp</code></a>、<a href="../_code/chapter05_generation/src/runtime.cpp"><code>runtime.cpp</code></a>、<a href="../_code/chapter05_generation/tests/sampler_test.cpp"><code>sampler_test.cpp</code></a>
- 相关阅读：[`05.7 端到端生成测试`](./07_end_to_end_generation.md)、[`章节导读`](./intro.md)

## 核心设计

模型前向只会产生分数，真正输出哪个 token 还需要采样策略决定。  
Sampler 是生成链路的最后一步之一，也最容易把“模型输出”和“系统决策”混在一起。

如果不把 `Sampler` 单独拎出来，调试时很容易把采样策略造成的差异误判成模型问题。

采样模块应当保持小而独立。仓库可执行实现是 `GreedySampler`；Temperature、Top-K 和 Top-p 用于解释后续接口设计，不属于当前测试覆盖。

最小伪代码可以写成：

```cpp
int sample_greedy(const Tensor& logits) {
    return argmax(logits);
}
```

Temperature 在哪里生效：

Temperature 不是在 softmax 之后再“调结果”，而是在 softmax 之前先缩放 logits：

```cpp
Tensor softmax_with_temperature(const Tensor& logits, float temp) {
    Tensor scaled = logits / temp;
    return softmax(scaled);
}
```

它的效果可以理解为：

- `temp > 1`：分布更平滑，随机性增加
- `temp = 1`：原始分布
- `temp < 1`：分布更锐利，确定性增加
- `temp -> 0`：趋近于 argmax，也就是 greedy

在实现里，temperature 通常是 sampler 的配置参数，而不是模型的一部分。

常用采样策略对比：

| 策略 | 操作 | 特点 | 常见用途 |
| --- | --- | --- | --- |
| Greedy | 选最高概率 token | 确定、可重复 | 基线 |
| Temperature | 缩放 logits | 控制随机性 | 所有采样模式 |
| Top-K | 从 K 个最高概率 token 中采样 | 候选集固定 | 限制随机范围 |
| Top-p | 从累计概率达到 p 的集合中采样 | 候选集动态 | 更自然的截断 |

Top-K 和 Top-p 怎么理解：

Top-K 的思路是：

1. 选出概率最高的 K 个 token
2. 只在这 K 个 token 中采样

Top-p 的思路是：

1. 按概率从高到低排序
2. 累积概率达到阈值 p 就停
3. 只在这个动态集合中采样

Top-K 的候选集大小固定，Top-p 的候选集会随分布变化。

如果把 sampler 放在生成链路的最后一步，就能清楚地区分两件事：

- 模型给了什么分数
- 系统最后选了什么 token

这对调试很重要，因为很多看起来像模型问题的现象，实际上只是采样策略不同。

`Greedy` 先把最简单的决策跑通。
等 `Greedy` 稳定后，再引入 `Top-K` 和 `Temperature`，就能更清楚地看见参数如何影响输出分布。

请求级采样配置可以设计成下面的接口形状，使不同请求采用不同随机性。该片段用于说明所有权，不是仓库现有 `Request` 的字段定义；当前代码只实现 `GreedySampler`，`Request` 保存 token history 和生命周期状态。

```cpp
struct SamplerConfig {
    float temperature;
    int top_k;
    float top_p;
};

struct GenerationRequest {
    KVCache cache_;
    SamplerConfig sampler_config_;
    std::vector<int> generated_;
};
```

实现请求级配置后，系统可以同时服务不同采样偏好的请求，比如：

- 一个请求走 greedy，追求确定性
- 另一个请求走 top-p，追求多样性

采样器位于生成循环的最后一步。下面是概念伪代码；仓库中的可执行入口是 `Runtime::generate(Request&, const Sampler&, RequestModelExecutor&)`。

```cpp
Sampler sampler(config);

while (!finished) {
    Tensor logits = model.forward(context);  // [vocab_size]
    int token = sampler.sample(logits);       // 采样决策
    context.push_back(token);
    if (token == eos_token) finished = true;
}
```

这样模型前向和采样决策在代码里是清楚分离的：模型负责给分数，sampler 负责把分数变成 token。

## 验证与衔接

现有测试重点看：

- 可以从 logits 中选出 token
- 最大 logit 对应的 token id 正确
- 相同 logits 的结果可重复

扩展 Temperature、Top-K 或 Top-p 时，再增加固定随机种子、候选集和分布边界测试。

常见问题：

- 把 logits 和概率混淆
- 温度参数方向理解错误
- 采样策略和缓存逻辑耦合太深

和后续章节怎么接：

`Sampler` 负责把分数变成 token。
下一页就可以把 embedding、Transformer、KV Cache、Scheduler 和 sampler 串成完整的端到端测试。
