# SGLang 前缀缓存与生产调度

## 为什么读

[SGLang](https://github.com/sgl-project/sglang) 同时覆盖推理 Runtime、结构化生成和服务系统。对本教程最有价值的是前缀 KV 复用怎样影响请求排序、缓存生命周期和资源预算，而不只是上层应用 workflow。

## 重点看什么

- RadixAttention/Radix Cache 怎样表示和复用可变长度共享前缀
- LPM 怎样把前缀命中信息用于等待队列排序
- Chunked Prefill 怎样和 token、KV、延迟预算共同工作
- HiCache 怎样把前缀 KV 扩展到 GPU、主机和外部存储层级

LPM 是调度策略，Radix Cache 是缓存与索引机制，HiCache 是多级存储扩展；三者相互协作，但不能写成同一个算法。本教程先保证请求级 K/V、block ownership 和释放语义正确，再讨论跨请求复用。

## 对应教程章节

- Chapter05：Prefill/Decode、Chunked Prefill、Scheduler 和 Block KV
- Chapter07-08：前缀命中 workload、TTFT 和资源成本
- Chapter09：admission、前缀复用、缓存回收和服务指标
