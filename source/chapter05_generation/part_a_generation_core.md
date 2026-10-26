# 05-A Generation Core 生成核心

Generation Core 跟踪一个请求从 prompt 到可重复 token 输出的完整路径。

## 阅读顺序

已有 Chapter04 或真实 Decoder 基础时，从生成循环开始，再用端到端测试和生命周期验证完整路径；理解 Chapter05 smoke model 时补读 Embedding 和 Transformer 组装。完成后应能追踪 `prefill -> decode -> sampler -> output`，并区分 token history 与请求级 Model Context。

| 顺序 | 类型 | 页面 | 观察对象 |
| --- | --- | --- | --- |
| 1 | 必读 | [`Naive Generation`](./03_naive_generation.md) | 逐 token 循环 |
| 2 | 必读 | [`KV Cache`](./04_kv_cache.md) | 三种历史状态与所有权 |
| 3 | 必读 | [`Scheduler`](./05_scheduler.md)、[`Sampler`](./06_sampler.md) | 推进策略与 token 决策 |
| 4 | 必读 | [`端到端生成测试`](./07_end_to_end_generation.md)、[`Request 生命周期`](./09_request_lifecycle.md) | 输出、终态与隔离 |
| 5 | 选读 | [`Embedding`](./01_embedding_and_position.md)、[`Transformer 组装`](./02_transformer_assembly.md) | 教学 smoke model 怎样提供 logits |

## 验证入口

```bash
bash code/chapter05_generation/scripts/run.sh
```

重点检查单请求生成、多请求隔离、取消/超时终态、token history、Model Context 归属和 streaming 输出。Chapter06-08 继续复用这里定义的 Request、Runtime、Sampler 和 executor 契约。

生成、终态和请求隔离均通过后，继续阅读 [`B. Serving Preview`](./part_b_serving_preview.md)，观察多个请求如何进入同一个持续运行的服务。
