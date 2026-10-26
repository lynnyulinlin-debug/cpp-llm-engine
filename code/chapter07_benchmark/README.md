# Chapter 07：Benchmark 基准测试与证据

本目录把正确性测试、阶段耗时、事件 trace、热点摘要和 JSON/Chrome trace 组织成可重复的 benchmark 入口。完整学习说明见 [`source/chapter07_benchmark`](../../source/chapter07_benchmark/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `src/benchmark.cpp` | benchmark 计时和阶段记录 |
| `src/test_helpers.cpp` | 固定输入和测试辅助 |
| `tests/chapter07_benchmark_test.cpp` | correctness 和 benchmark 输出测试 |
| `examples/main_benchmark.cpp` | 生成阶段、事件和 trace 示例 |
| `examples/runtime_benchmark.cpp` | Chapter 05 CPU runtime 实测 benchmark |
| `examples/multi_request_benchmark.cpp` | `1/2/4` 请求场景和 JSON 结果输出 |
| `scripts/run.sh` | 隔离构建、CTest 和 benchmark 入口 |
| `results/chapter07_cpu_baseline.json` | 最近一次可复查的 CPU 多请求基线结果 |
| Chapter 08 | 消费本章 trace 和 hotspot 证据进行 profiling |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter07_benchmark/scripts/run.sh
```

`runtime_benchmark` 输出的是 Chapter 05 CPU runtime 的实测结果；`main_benchmark` 中的固定 trace 只用于演示结果格式。比较结果时固定输入、设备、编译选项和脚本版本；benchmark 输出用于支持决策，不替代正确性测试。

`multi_request_benchmark` 使用同一个 CPU Runtime 和 Round-robin Scheduler，依次测量 1、2、4 个请求。脚本默认使用 warmup=1、iterations=3，并将结果写入 `code/chapter07_benchmark/results/chapter07_cpu_baseline.json`。耗时用于观察趋势，`correctness`、请求完成数和生成 token 数才是本轮的硬性验收条件。
