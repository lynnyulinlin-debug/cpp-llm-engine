# 05-A Generation Core 生成核心

这一部分只回答一个问题：一个请求怎样从 prompt 走到可重复的 token 输出。

## 目标

完成后应能运行并验证：

| 能力 | 产物 |
| --- | --- |
| 输入进入模型 | Embedding、位置编码和 Transformer 组装 |
| 逐 token 生成 | `prefill -> decode -> sampler -> output` |
| 保存历史状态 | Request token history 与请求级 Model Context |
| 推进请求 | 基础 Scheduler 和终态判断 |
| 输出结果 | 普通生成和 streaming callback |

## 阅读顺序

已有 Chapter04 或真实 Decoder 基础时，不需要先读教学 Transformer 组装。首轮从生成循环开始，最后用端到端测试和生命周期收口；需要理解 Chapter05 smoke model 时再补前两页。

| 顺序 | 类型 | 页面 | 观察对象 |
| --- | --- | --- | --- |
| 1 | 必读 | [`Naive Generation`](./03_naive_generation.md) | 逐 token 循环 |
| 2 | 必读 | [`KV Cache`](./04_kv_cache.md) | 三种历史状态与所有权 |
| 3 | 必读 | [`Scheduler`](./05_scheduler.md)、[`Sampler`](./05_sampler.md) | 推进策略与 token 决策 |
| 4 | 必读 | [`端到端生成测试`](./06_end_to_end_generation.md)、[`Request 生命周期`](./08_request_lifecycle.md) | 输出、终态与隔离 |
| 5 | 选读 | [`Embedding`](./01_embedding_and_position.md)、[`Transformer 组装`](./02_transformer_assembly.md) | 教学 smoke model 怎样提供 logits |

## 验证入口

```bash
bash code/chapter05_generation/scripts/run.sh
```

重点检查单请求生成、多请求隔离、取消/超时终态、token history、Model Context 归属和 streaming 输出。Chapter06-08 继续复用这里定义的 Request、Runtime、Sampler 和 executor 契约。

## 进入下一部分

当 Generation Core 已经能够稳定生成后，再阅读 [`B. Serving Preview`](./part_b_serving_preview.md)，理解多个请求如何进入同一个持续运行的服务。
