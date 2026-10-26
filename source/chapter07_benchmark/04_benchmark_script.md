# 07.4 Benchmark Script Benchmark 脚本


## 把 Benchmark 变成脚本

编写 `benchmark.sh`，自动运行 benchmark 并记录硬件与运行信息。

读完后应能做到：

- 理解为什么 benchmark 需要自动化
- 理解应该记录哪些环境信息
- 理解脚本如何支持重复测量

**阅读入口**

- 前置阅读：[`07.3 Benchmark Metrics`](./03_benchmark_metrics.md)、[`07.2 End-to-End Correctness Testing`](./02_e2e_correctness.md)
- 源码直达：<a href="../_code/chapter07_benchmark/scripts/run.sh"><code>run.sh</code></a>、<a href="../_code/chapter07_benchmark/examples/main_benchmark.cpp"><code>main_benchmark.cpp</code></a>、<a href="../_code/chapter07_benchmark/tests/chapter07_benchmark_test.cpp"><code>chapter07_benchmark_test.cpp</code></a>
- 相关阅读：[`07.6 Result Interpretation`](./06_result_interpretation.md)、[`回归测试`](../reference/regression_testing.md)

## 脚本设计

没有脚本化的 benchmark 很难复现，也很难比较不同版本的性能。

benchmark 脚本需要覆盖：

- 自动运行测试
- 记录硬件信息
- 保存运行结果
- 输出可比较的 benchmark 报告

脚本化 benchmark 的价值在于把“运行条件”固定下来。没有固定输入、固定模型、固定环境信息，测出来的性能数据很难真正比较。

因此脚本不仅要跑测试，还要把版本、硬件和配置一起写进结果里。

建议脚本输出同时包含两种格式：一种给人看，另一种给后续自动分析用。只留文本日志的话，后面做回归比较会很费劲。

脚本流程可以先写成：

```bash
#!/usr/bin/env bash
set -euo pipefail
./run_tests
./run_benchmark > benchmark.json
```

在本教程中，`run_tests` 和 `run_benchmark` 对应的是 `code/chapter07_benchmark/scripts/run.sh` 管理的隔离构建目录。脚本会运行 Chapter 05 CPU runtime benchmark、Chapter 07 多请求 benchmark 和 CTest，并将多请求结果写入 `code/chapter07_benchmark/results/chapter07_cpu_baseline.json`。

实现建议：

- 让脚本能独立执行
- 固定输入和配置
- 把环境信息和结果一起保存

多请求最小场景固定为 `1/2/4` 个请求，默认 warmup=1、iterations=3。结果文件记录请求数、prefill/decode/total 时间、TTFT、token throughput、完成请求数和 correctness，便于后续 Chapter 08 使用同一输入条件定位热点。

Chapter07 默认脚本使用教学 Runtime，适合快速回归 benchmark 结构。完成真实 TinyLlama Model/Context 接入后，可运行 `bash code/engine/scripts/run_real_serving_benchmark.sh`；该脚本固定两个真实请求，记录 CPU 型号、编译器、构建类型、模型加载、prefill/decode、每请求 TTFT/完成时间和吞吐，并把结果交给 [`09.3 真实 ServiceCore Benchmark`](../chapter09_serving/03_real_service_benchmark.md) 解读。真实模型 benchmark 不进入默认回归，避免让 638 MiB GGUF 成为基础测试依赖。

## 验证与排查

验证时重点看：

- 脚本能自动完成一次测量
- 硬件信息被正确记录
- 结果可重复生成

如果同一脚本在同一环境里多次运行，输出结果格式稳定且便于横向比较，benchmark 脚本就达标了。

常见问题：

- 测量条件没有固定
- 输出结果没有归档
- 硬件信息采集不完整
