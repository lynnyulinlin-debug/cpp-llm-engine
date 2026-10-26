# 08.4 Kernel Launch and CUDA Graph Kernel 启动开销

Decode 会反复提交形状较小的 kernel。当单次计算很短时，host 提交、同步和 kernel 间空隙可能成为主要成本。本节产物是一套 CUDA Graph 决策方法：先证明固定开销存在，再判断计算图是否稳定到可以捕获和复用。

**阅读入口**

- 前置阅读：[`03.4 cuBLAS Backend Design`](../chapter03_matmul/04_cublas_implement.md)、[`08.2 Latency Breakdown`](./02_latency_breakdown.md)
- 当前证据：<a href="../_code/chapter07_benchmark/src/benchmark.cpp"><code>benchmark.cpp</code></a> 输出阶段、事件和 hotspot；<a href="../_code/cuda/scripts/run.sh"><code>code/cuda/scripts/run.sh</code></a> 验证已有 CUDA kernel
- 相关阅读：[`foundation/07 CUDA 常用库`](../foundation/07_cuda_common_libraries.md)、[`08.5 A/B Experiment`](./05_ab_experiment.md)

## 先判断瓶颈类型

CUDA Graph 优化重复提交稳定计算图的固定开销。若时间主要落在大矩阵或 Attention kernel 内部，应先优化对应 kernel；若时间线中存在大量短 kernel、host 提交和同步空隙，再进入 Graph 或 fusion 实验。

| 观察结果 | 优先动作 |
| --- | --- |
| GEMM / Attention kernel 占据大部分时间 | 调整 kernel、layout、dtype 或 backend |
| 大量短 kernel，中间有明显空隙 | 评估 CUDA Graph 或 kernel fusion |
| Host-device 同步密集 | 减少同步和数据往返 |
| Decode shape、地址或控制流频繁变化 | 先稳定 runtime 与 buffer 生命周期 |
| Decode step 形状稳定且重复次数多 | 建立 Graph capture/replay A/B |

## 当前教程能观察什么

Chapter07 的 synthetic trace 把一次请求拆成阶段、事件和 hotspot。它用于练习归因顺序，不是 GPU kernel trace。学习者先确认最重事件属于 prefill、decode 还是输出，再决定需要哪种底层工具。

| 证据视图 | 当前可回答的问题 | 进入真实 GPU 后补充什么 |
| --- | --- | --- |
| phase summary | prefill、decode、sampling、output 谁占主导 | CPU/GPU 阶段重叠 |
| trace events | `decode_step` 与 `sample_logits` 是否分开 | kernel launch、stream、memcpy 和同步事件 |
| hotspot table | 应先调查哪个函数或事件 | kernel self time、调用次数和占用率 |
| Chrome trace JSON | host/compute 时间线如何排列 | CUDA stream、correlation ID 和 device duration |

生成当前结构化证据：

```bash
bash code/chapter07_benchmark/scripts/run.sh
```

运行现有 GPU correctness 基线：

```bash
bash code/cuda/scripts/run.sh
```

## CUDA Graph 实验契约

真实实验需要固定 decode workload，并分别采集普通 launch 与 Graph replay。两组使用相同模型、token 数、batch、stream、buffer 地址和 correctness golden；报告至少记录端到端 decode、device compute、launch/sync 空隙、capture 成本和 replay 次数。

Graph 路径还要验证三类变化：batch 或 sequence shape 改变时选择正确 graph；请求终止后 buffer 仍由正确对象持有；fallback 路径在无法 replay 时保持相同输出。把这些条件写进实验记录后，再实现 capture cache 和 replay dispatch。

| 判定 | 条件 |
| --- | --- |
| `KEEP` | correctness 通过，稳态 decode 改善超过阈值，capture 成本能被 replay 次数摊销 |
| `REVERT` | 输出变化、资源生命周期不稳定，或稳态收益低于阈值 |
| `SPLIT` | launch overhead 已确认，但 Graph 与 fusion 同时修改，无法归因 |

## 本节完成标准

完成后应能从 timeline 区分 kernel compute、launch gap 和同步等待；能说明当前 synthetic trace 与真实 CUDA trace 的证据差异；能为固定 decode workload 写出单变量 Graph A/B，而不是根据 kernel 数量直接假定 CUDA Graph 有收益。
