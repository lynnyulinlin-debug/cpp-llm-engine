# 09.3 真实 ServiceCore Benchmark

真实双请求正确性通过后，下一步不是立即优化，而是固定输入、环境和输出格式，记录可重复的 CPU 基线。本节测量 `RealLlamaExecutorFactory -> ServiceCore -> Scheduler -> Runtime`，不经过 HTTP transport，因此结果反映模型执行与运行时编排，不包含网络延迟。

从仓库根目录执行：

```bash
bash code/engine/scripts/run_real_serving_benchmark.sh
```

脚本默认使用仓库中的 TinyLlama GGUF，按 `warmup=1`、`iterations=3` 构建 Release 目标，并把结果写入 <a href="../_code/chapter09_serving/results/tinyllama_service_cpu_baseline.json"><code>tinyllama_service_cpu_baseline.json</code></a>。可以通过 `REAL_SERVING_WARMUP`、`REAL_SERVING_ITERATIONS`、`REAL_SERVING_BUILD_DIR` 和 `REAL_SERVING_RESULT_FILE` 覆盖默认值。

## 固定条件与指标

benchmark 同时提交 `Hello TinyLlama` 和 `Hi`，共享一份 22 层 Model，每个请求创建独立 Context，各生成 4 个 token。计时装饰器包裹 `RequestModelExecutor`，因此不修改 Decoder、Runtime 或 ServiceCore 的行为。

| 记录项 | 计时范围 | 用途 |
| --- | --- | --- |
| tokenizer/model load | 进程启动后的一次性加载 | 区分冷启动与稳态请求 |
| prefill compute | 两个 Executor 的 `prefill()` 累计 | 观察 prompt 处理成本 |
| decode compute | 六次 `decode()` 累计 | 观察后续 token 成本 |
| runtime overhead | dispatch 减去 prefill/decode | 估算 Scheduler、Sampler 和状态更新成本 |
| request TTFT | dispatch 到各请求首 token callback | 比较队列顺序对首 token 的影响 |
| request completion | dispatch 到各请求最后一个 token | 比较 Round Robin 下的完成时间 |
| throughput | 8 个生成 token / dispatch 时间 | 记录固定场景的稳态产出 |
| correctness | golden token、cache 隔离、Context 释放 | 性能数据进入比较前的硬闸门 |

## 离线与在线口径

离线 benchmark 保留每个请求的独立样本，适合固定环境下计算 P50 和范围；在线 `/metrics` 使用累计 `_sum/_count`，避免 ServiceCore 保存无界样本。两者都使用单调时钟，但起点不同：离线 TTFT 从 `dispatch` 起算，在线 TTFT 从 `admission` 起算，因此在线值包含 queue wait，不能把两组数字直接相减。

| 指标 | 离线 benchmark | 在线 `/metrics` |
| --- | --- | --- |
| Queue wait | 当前固定场景接近零，未单独归档 | admission 到进入 running |
| TTFT | dispatch 到首 token | admission 到首 token |
| Completion | dispatch 到最后 token | admission 到 terminal |
| 聚合方式 | 每轮样本、mean/P50/min/max | `_sum/_count`，由监控系统计算均值 |

## 当前 CPU 基线

当前结果来自 `AMD Ryzen 9 9955HX`、GCC 13.1.0、Release 构建。数字只用于同一机器、同一模型、同一输入和相同编译条件下的后续 A/B 对比，不与 vLLM、SGLang、TensorRT-LLM 或其他硬件直接横比。

三轮结果显示，双请求 dispatch 平均约 `7229.15 ms`，吞吐约 `1.107 generated token/s`。prefill 与 decode 的模型计算合计约 `7229.00 ms`，Runtime/Scheduler 平均开销约 `0.150 ms`。这组数据把瓶颈指向 CPU Decoder kernel 与串行模型执行，而不是 ServiceCore 队列操作；优化实验应先用 Chapter08 profiling 验证模型热点，不应直接重写 admission。

| 指标 | Mean | P50 | 范围 |
| --- | ---: | ---: | ---: |
| Prefill compute | `3797.11 ms` | `3797.28 ms` | `3789.23-3804.82 ms` |
| Decode compute | `3431.89 ms` | `3431.58 ms` | `3426.11-3437.98 ms` |
| Runtime overhead | `0.150 ms` | `0.148 ms` | `0.146-0.156 ms` |
| Dispatch | `7229.15 ms` | `7235.41 ms` | `7215.48-7236.56 ms` |
| Request A TTFT | `3226.77 ms` | `3228.88 ms` | `3217.70-3233.71 ms` |
| Request B TTFT | `3797.15 ms` | `3797.32 ms` | `3789.26-3804.86 ms` |
| Throughput | `1.107 token/s` | `1.106 token/s` | `1.105-1.109 token/s` |
| Correctness | `PASS` | `PASS` | 三轮全部通过 |

## 组合 optimized policy

Chapter08 完成三组单变量实验后，通过 `RealLlamaExecutorFactory` 的显式 `optimized-avx2` 模式把 Q4_K Attention、Q4_K FFN 和 Q6_K FFN backend 应用到每个请求 Context，LM Head 保持 scalar。默认服务和原始 baseline 不变；组合实验由下面的命令单独运行：

```bash
bash code/chapter08_profiling/scripts/run_real_optimized_policy_ab.sh
```

比较器要求两组运行使用相同模型、CPU、编译器、Release 构建、prompt、请求数、token budget、warmup 和迭代数，并同时通过 golden token、Context/cache 隔离与 terminal release。

| 指标 | Scalar | Optimized AVX2 | Delta |
| --- | ---: | ---: | ---: |
| Decoder 总耗时 | `5208.81 ms` | `1078.23 ms` | `-79.2998%` |
| ServiceCore dispatch | `7573.39 ms` | `1629.51 ms` | `-78.4838%` |
| Prefill compute | `3977.48 ms` | `777.73 ms` | `-80.4467%` |
| Decode compute | `3595.75 ms` | `851.62 ms` | `-76.3160%` |
| Throughput | `1.0564 token/s` | `4.9095 token/s` | `+364.7483%` |
| Correctness | `PASS` | `PASS` | `KEEP` |
