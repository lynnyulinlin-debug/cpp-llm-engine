# 07.7 主流框架 Benchmark 对照

不同框架的默认 workload、服务方式和指标口径并不相同。对照时先固定模型、输入输出长度、并发、硬件和正确性，再比较结果。

## 指标映射

| 教程证据 | 成熟 benchmark 中的对应项 | 先检查什么 |
| --- | --- | --- |
| correctness gate | golden output、模型质量或请求成功率 | 性能结果是否建立在正确输出上 |
| prefill / decode 时间 | TTFT、TPOT、ITL | 指标起止点是否一致 |
| tokens/s | request/output throughput | 分母是总时间、稳态时间还是模型时间 |
| 双请求时间轴 | concurrency / serving workload | 请求到达方式和调度策略 |
| JSON 结果文件 | benchmark report | 模型、dtype、硬件和命令是否可复现 |

## 教学实现与生产实现

| 本章可验证对象 | 成熟系统继续增加 | 后续入口 |
| --- | --- | --- |
| correctness-first、阶段耗时和吞吐 | 大规模并发负载生成 | Chapter09 服务压测扩展 |
| 固定双请求 benchmark | 到达率分布和长时间稳态 | 服务 benchmark 扩展 |
| 机器可读 JSON | 结果数据库和趋势面板 | CI/持续回归 |
| CPU 本机对照 | 多 GPU、多节点比较 | GPU/分布式主线 |

## 结果阅读问题

1. 两组结果的模型、dtype、输入输出长度、并发和硬件是否一致？
2. TTFT、TPOT、throughput 的起止点和聚合方式是否一致？
3. 优化结果是否附带 correctness、失败率和资源占用证据？

| 项目卡 | 在本章中的用途 |
| --- | --- |
| [llama.cpp](../reference/ecosystem/llama_cpp.md) | 本地 prompt processing 与 generation 基线 |
| [vLLM](../reference/ecosystem/vllm.md) | online、offline 和 latency workload |
| [SGLang](../reference/ecosystem/sglang.md) | serving 与 cache workload |
| [TensorRT-LLM](../reference/ecosystem/tensorrt_llm.md) | Engine 配置与 GPU Benchmark |
