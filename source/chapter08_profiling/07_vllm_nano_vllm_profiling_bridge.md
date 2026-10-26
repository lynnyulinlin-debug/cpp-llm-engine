# 08.7 成熟框架 Profiling 阅读桥

本章用阶段、事件和 hotspot 三层证据定位瓶颈。成熟框架会换成更完整的 trace、GPU profiler 和在线指标，但归因顺序不变。

## 证据映射

| 教程证据 | 成熟工具中的对应物 | 先判断什么 |
| --- | --- | --- |
| `prefill` / `decode` 阶段时间 | range、span、engine phase metric | 慢发生在哪个阶段 |
| `trace_events` | Chrome trace、Nsight Systems timeline | 计算、同步和 transport 是否重叠 |
| `hotspots` | CPU sample、CUDA kernel summary | 哪个函数或 kernel 值得下钻 |
| `tid` / `sync_*` | host thread、CUDA stream、同步事件 | 时间花在计算还是等待 |
| correctness-first A/B | optimization regression report | 单变量改动是否值得保留 |

## 教学实现与生产实现

稳定 workload 和 correctness gate 先复现性能现象，Profiler 再解释阶段、事件和热点。

| 本章可验证对象 | 成熟系统继续增加 | 使用条件 |
| --- | --- | --- |
| 阶段 JSON、最小 trace 和 hotspot | 完整 CPU sampling | 获得 `perf` 权限后补充 |
| host/compute timeline 和同步点 | CUDA kernel 指标 | GPU backend 建立后使用 Nsight |
| Decoder/ServiceCore 单变量 A/B | 分布式 trace | 多 worker 主线建立后补充 |
| KEEP/ROLLBACK 判定 | 自动性能回归告警 | 接入稳定 CI 基线后补充 |

## Profiling 阅读问题

1. benchmark 现象能否稳定复现，correctness 是否先通过？
2. 热点属于模型计算、cache、scheduler、同步还是 transport？
3. 修改后同一 workload 的阶段、事件和 hotspot 是否共同支持结论？

| 项目卡 | 在本章中的用途 |
| --- | --- |
| [llama.cpp](../reference/ecosystem/llama_cpp.md) | CPU/GPU backend 的本地基线 |
| [vLLM](../reference/ecosystem/vllm.md) | Benchmark 指标到 EngineCore 阶段的映射 |
| [TensorRT-LLM](../reference/ecosystem/tensorrt_llm.md) | Engine、Runtime 和 GPU kernel 时间关联 |
