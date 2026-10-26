# vLLM 高吞吐推理引擎

## 为什么读

[vLLM](https://github.com/vllm-project/vllm) 适合观察模型执行、请求调度和 KV 内存管理怎样组成高吞吐引擎。阅读重点不是某个 kernel，而是 EngineCore、scheduler、worker/model runner 和 cache manager 之间的职责。

## 重点看什么

- Continuous Batching 怎样按 iteration 重组运行请求
- Block/Paged KV 怎样把逻辑 token 位置映射到物理缓存
- Prefill 与 Decode 怎样共享执行器又保持不同 workload
- Prefix caching、抢占和恢复需要增加哪些状态与资源约束

本教程用 Chapter05 的 CPU Runtime 固定请求推进语义，用 Chapter09 固定服务生命周期；不追求按 vLLM 的目录和进程模型做 1:1 复刻。

## 对应教程章节

- Chapter04：模型层、Attention backend 和权重映射
- Chapter05：Request、Scheduler、Continuous Batching 和 Block KV
- Chapter07-08：服务 workload、吞吐/延迟指标和 profiling
- Chapter09：EngineCore、worker 与服务入口的职责边界
