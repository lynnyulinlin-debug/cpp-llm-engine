# 05.6 End-to-End Generation Testing 端到端生成测试


## 这一页要解决什么

完成一次端到端生成测试：输入 prompt，输出固定数量的 token，确认生成链路完整可用。

## 阅读入口

- 前置阅读：[`05.4 KV Cache`](/chapter05_generation/04_kv_cache)、[`05.5 Sampler`](/chapter05_generation/05_sampler)
- 源码直达：[`chapter05_generation 代码入口`](/code/chapter05_generation)
- 相关阅读：[`05.7 本章小结`](/chapter05_generation/summary)、[`chapter06_quantization`](/chapter06_quantization/intro)

## 2. 背景是什么

前面的模块分别正确，并不代表组合起来也正确。端到端测试是最终验收点。

这一页要确认的是：输入能进、状态能走、cache 能复用、sampler 能出 token，最后还能稳定回到一个可检查的输出。

## 3. 最小设计是什么

最小测试只需要固定两件事：

- prompt
- 生成 token 数量

这样就能把它当成一条稳定的回归基线，而不是一次性的手工演示。

### 端到端测试的成功标准

| 标准 | 说明 |
| --- | --- |
| 无崩溃 | 生成过程中没有内存错误或异常 |
| 输出长度正确 | 生成了 `max_new_tokens` 个 token |
| 采样可复现 | 在 greedy 模式下，多次运行结果一致 |
| 非空输出 | 至少生成了一个 token |
| 停止条件触发 | 达到 `max_new_tokens` 或生成 EOS |

最低通过标准是：输出非空且长度正确。  
进阶标准是：在 greedy 模式下，输出与记录过的黄金输出完全一致。

### 黄金输出是什么

黄金输出（golden output）是用于回归测试的稳定参考：

```cpp
// 第一次运行：记录输出
std::string prompt = "Hello";
auto output = generate(prompt, 16);
std::string expected = output;

// 后续运行：对比输出
auto new_output = generate(prompt, 16);
EXPECT_EQ(new_output, expected);
```

注意：

- 黄金输出最适合 greedy 模式，因为它是确定性的
- 如果使用 temperature 采样，通常要固定 seed 才能复现
- 模型更新后，黄金输出可能需要重新生成

## 4. 中间推导是什么

端到端测试之所以要放在 KV Cache 之后，是因为它验证的不是单个模块，而是整个 runtime contract：

- embedding 是否和 decoder 对齐
- cache 是否真的复用了历史状态
- sampler 是否把 logits 变成了下一步 token
- 输出数量和停止条件是否可控

如果这里失败，通常需要回头检查 embedding、decoder、cache 或 sampler 其中某一段。

最小测试流程可以写成：

```cpp
std::string prompt = "Hello";
auto output = generate(prompt, /*max_new_tokens=*/16);
EXPECT_FALSE(output.empty());
EXPECT_EQ(output.size(), 16);
```

这里的重点是“链路闭环”，不是生成质量。

### Sampler 的集成验证

Sampler 的验证通常可以通过 greedy 模式来做：

```cpp
auto config = Config();
config.sampler.temperature = 0.0f;  // greedy
config.sampler.top_k = 0;           // 禁用 top-k

auto output1 = generate("Hello", config, 16);
auto output2 = generate("Hello", config, 16);
EXPECT_EQ(output1, output2);
```

如果两次输出不一致，说明 sampler 里还有非确定性因素，比如随机种子没有固定。

## 5. 怎么验证

- 程序可以稳定生成 token
- 输出数量达到预期
- 没有在中途崩溃或卡死
- 同一 prompt 下，结果能稳定复现
- greedy 模式下可以得到黄金输出或完全一致的回归结果

一个好的端到端测试应当能在固定输入下稳定复现同类行为，这样后续重构时才有可比基线。

### 失败时怎么排查

端到端测试失败时，可以按这个顺序往回查：

1. 检查输出长度
   - 长度 = 0，说明生成循环没真正跑起来
   - 长度 < 预期，说明可能提前遇到 EOS
   - 长度 > 预期，说明停止条件没触发
2. 检查 cache
   - 单独对比 naive 和 cache 版本的输出
3. 检查 sampler
   - 确认 logits 的 shape 是 `[vocab_size]`
   - 确认 greedy 模式下是确定性的
4. 检查 embedding 和 decoder 的 shape 一致性
   - embedding 输出应是 `[B, S, D]`
   - decoder 输入也应是 `[B, S, D]`

## 6. 常见问题

- 生成数量配置错误
- prompt 编码错误
- 前面某个模块的 shape 问题在这里被放大
- 只验证“有输出”，没有验证“输出长度和停止条件是否受控”
- 把端到端测试当成质量评估，而不是链路闭环检查
- 一旦结果异常，无法判断问题是出在 cache 还是 sampler

## 7. 和前面的关系

这一页不是独立的演示页，而是这一章的收口动作：

- `01_embedding_and_position` 负责输入如何进入模型
- `04_kv_cache` 负责历史状态如何保存和复用
- 这一页负责把前面的所有步骤串成可回归的闭环

如果这一页不稳定，说明 generation runtime 还不能算真正闭合。
